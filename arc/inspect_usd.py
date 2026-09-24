"""Inspect a robot USD on the LOGIN NODE -- no GPU, no Isaac Sim, no queue.

Prints what configs/robot.yaml needs for a differential-drive robot: joint
names + drive gains, rigid bodies + masses, wheel radius, wheel separation,
footprint radius, and how far the root sits above the robot's lowest point.

One-time setup (a throwaway venv; NOT a project dependency):
    python3 -m venv ~/usdtools && ~/usdtools/bin/pip install usd-core

Run:
    ~/usdtools/bin/python arc/inspect_usd.py ~/isaac_assets/carter/carter_v1.usd
"""

from __future__ import annotations

import math
import sys

from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils


def _bbox(cache: UsdGeom.BBoxCache, prim: Usd.Prim):
    box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return None if box.IsEmpty() else (box.GetMin(), box.GetMax())


def main(path: str) -> None:
    # ---- missing sub-files (a referenced .usd that was not downloaded) -------
    _layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(path)
    print(f"file: {path}")
    if unresolved:
        print("UNRESOLVED dependencies (download these next to the file):")
        for u in unresolved:
            print(f"   {u}")
    else:
        print("dependencies: all resolved")

    stage = Usd.Stage.Open(path)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    print(f"metersPerUnit={mpu}  upAxis={UsdGeom.GetStageUpAxis(stage)}")
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    print(f"default prim: {root.GetPath()}")

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.guide],
        useExtentsHint=True,
    )
    xf = UsdGeom.XformCache()

    # ---- articulation root, rigid bodies, masses ------------------------------
    print("\n-- articulation root / rigid bodies (mass kg) --")
    bodies = {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            print(f"  ARTICULATION ROOT: {prim.GetPath()}")
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get() if prim.HasAPI(UsdPhysics.MassAPI) else None
            bodies[prim.GetName()] = prim
            print(f"  {prim.GetName():<28} mass={mass}")

    # ---- joints + drives -------------------------------------------------------
    print("\n-- joints (type, axis, bodies, drive) --")
    wheel_joints = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(prim)
        kind = prim.GetTypeName()
        b0 = [str(t) for t in j.GetBody0Rel().GetTargets()]
        b1 = [str(t) for t in j.GetBody1Rel().GetTargets()]
        axis = prim.GetAttribute("physics:axis").Get() if prim.HasAttribute("physics:axis") else "-"
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        dstr = "none"
        if drive:
            dstr = (
                f"type={drive.GetTypeAttr().Get()} stiffness={drive.GetStiffnessAttr().Get()} "
                f"damping={drive.GetDampingAttr().Get()} maxForce={drive.GetMaxForceAttr().Get()}"
            )
        print(f"  {prim.GetName():<24} {kind:<22} axis={axis}  {b0} -> {b1}\n      drive: {dstr}")
        if kind == "PhysicsRevoluteJoint" and "wheel" in prim.GetName().lower():
            wheel_joints.append((prim.GetName(), b1[0] if b1 else ""))

    # ---- geometry: footprint, root height, wheels ------------------------------
    print("\n-- geometry (metres) --")
    rb = _bbox(cache, root)
    if rb:
        lo, hi = rb
        dx, dy, dz = (hi[i] - lo[i] for i in range(3))
        radius = 0.5 * math.hypot(dx, dy) * mpu
        root_z = xf.GetLocalToWorldTransform(root).ExtractTranslation()[2]
        print(f"  overall size       : {dx * mpu:.3f} x {dy * mpu:.3f} x {dz * mpu:.3f}")
        print(f"  footprint radius   : {radius:.3f}  (half-diagonal of the x-y extent)")
        print(f"  root above lowest  : {(root_z - lo[2]) * mpu:.3f}  (spawn height must be >= this)")

    centers = {}
    for name, body_path in wheel_joints:
        prim = stage.GetPrimAtPath(body_path)
        if not prim:
            continue
        box = _bbox(cache, prim)
        if box:
            lo, hi = box
            centers[name] = [(lo[i] + hi[i]) / 2 * mpu for i in range(3)]
            print(f"  wheel {name:<18}: radius ~{(hi[2] - lo[2]) / 2 * mpu:.4f}  center={['%.3f' % c for c in centers[name]]}")
    if len(centers) >= 2:
        (n1, c1), (n2, c2) = list(centers.items())[:2]
        print(f"  wheel separation   : {math.dist(c1, c2):.4f}  ({n1} <-> {n2})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: inspect_usd.py <robot.usd>")
    main(sys.argv[1])
