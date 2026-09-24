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

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdUtils


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

    # Real points only (useExtentsHint=False): authored extent hints on these
    # assets can be stale or in other units, which produced a "3 km" robot.
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
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

    # ---- collision geometry: what physics actually touches -------------------
    # Extents are COMPUTED from each shape's own attributes (Cube.size,
    # Cylinder.radius/height, mesh points) via ComputeExtentFromPlugins, then
    # the 8 corners are transformed to world. Authored `extent` attributes on
    # these assets are stale (a 0.48 m wheel reported as 55 m), so neither
    # BBoxCache nor extentsHint can be trusted here. Only prims under the
    # default prim count -- that is all Isaac Lab references into the scene.
    print("\n-- collision shapes (metres, computed from shape attributes) --")
    root_path = root.GetPath()
    world = {}
    true_radius = {}
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not prim.GetPath().HasPrefix(root_path):
            continue
        boundable = UsdGeom.Boundable(prim)
        ext = UsdGeom.Boundable.ComputeExtentFromPlugins(boundable, Usd.TimeCode.Default()) if boundable else None
        if not ext:
            print(f"  (no extent) {prim.GetPath()}")
            continue
        m = xf.GetLocalToWorldTransform(prim)
        corners = [m.Transform(Gf.Vec3d(x, y, z)) for x in (ext[0][0], ext[1][0])
                   for y in (ext[0][1], ext[1][1]) for z in (ext[0][2], ext[1][2])]
        lo = [min(c[i] for c in corners) * mpu for i in range(3)]
        hi = [max(c[i] for c in corners) * mpu for i in range(3)]
        body = prim
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        bname = body.GetName() if body else "?"
        world.setdefault(bname, []).append((lo, hi))
        size = [hi[i] - lo[i] for i in range(3)]
        center = [(hi[i] + lo[i]) / 2 for i in range(3)]
        extra = ""
        # A ROTATED cylinder's bounding box overstates its radius (the earlier
        # 0.278/0.271 "wheel radius" was exactly that). Read it from the shape:
        # radius attr x the world scale of a direction perpendicular to its axis.
        if prim.IsA(UsdGeom.Cylinder) or prim.IsA(UsdGeom.Capsule):
            axis = prim.GetAttribute("axis").Get() or "Z"
            perp = Gf.Vec3d(0, 0, 1) if axis in ("X", "Y") else Gf.Vec3d(1, 0, 0)
            r_world = float(prim.GetAttribute("radius").Get()) * m.TransformDir(perp).GetLength() * mpu
            true_radius[bname] = r_world
            extra = f"  radius={r_world:.4f} (attr x scale, axis {axis})"
        print(f"  {bname:<18} {prim.GetTypeName():<9} size={['%.3f' % v for v in size]} "
              f"center={['%.3f' % v for v in center]}{extra}")

    allb = [b for boxes in world.values() for b in boxes]
    if allb:
        lo = [min(b[0][i] for b in allb) for i in range(3)]
        hi = [max(b[1][i] for b in allb) for i in range(3)]
        dx, dy, dz = (hi[i] - lo[i] for i in range(3))
        root_z = xf.GetLocalToWorldTransform(root).ExtractTranslation()[2] * mpu
        print("\n-- derived (collision geometry) --")
        print(f"  collision size     : {dx:.3f} x {dy:.3f} x {dz:.3f}")
        print(f"  x range / y range  : [{lo[0]:.3f}, {hi[0]:.3f}] / [{lo[1]:.3f}, {hi[1]:.3f}]")
        print(f"  footprint radius   : {math.hypot(max(abs(lo[0]), abs(hi[0])), max(abs(lo[1]), abs(hi[1]))):.3f}"
              "  (farthest x-y corner from the root -- the robot turns about the root)")
        print(f"  root above lowest  : {root_z - lo[2]:.3f}  (spawn height must be >= this)")

    wc = {}
    for name, body_path in wheel_joints:
        prim = stage.GetPrimAtPath(body_path)
        boxes = world.get(prim.GetName(), []) if prim else []
        if boxes:
            lo = [min(b[0][i] for b in boxes) for i in range(3)]
            hi = [max(b[1][i] for b in boxes) for i in range(3)]
            wc[name] = [(lo[i] + hi[i]) / 2 for i in range(3)]
            r = true_radius.get(prim.GetName())
            r_txt = f"{r:.4f} (from shape attributes)" if r is not None else f"~{(hi[2] - lo[2]) / 2:.4f} (bbox; unreliable if rotated)"
            print(f"  wheel {name:<12}: radius {r_txt}  center={['%.3f' % c for c in wc[name]]}")
    if len(wc) >= 2:
        (n1, c1), (n2, c2) = list(wc.items())[:2]
        print(f"  wheel separation   : {math.dist(c1, c2):.4f}  (collision-cylinder centres {n1} <-> {n2})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: inspect_usd.py <robot.usd>")
    main(sys.argv[1])
