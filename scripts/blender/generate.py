"""
Synthetic Glass & Liquid Dataset Generator (Cycles Edition)
============================================================
Runs headless inside Blender's bundled Python (bpy).
Uses Cycles renderer for physically correct glass transparency.
Generates YOLO-format bounding box labels.

Domain randomization:
  1. Liquid color palettes (water, coffee, juice, milk)
  2. Ice cubes (30% of frames)
  3. Multiple HDRIs (scanned from assets/hdri/)
  4. Depth of field (f/1.4 – f/8.0)
  5. Grain & post-processing (compositor noise overlay)

Usage:
    tools/blender-portable/blender.exe --background --python scripts/blender/generate.py [-- --num-images 5000]
"""

import json
import math
import os
import random
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT_DIR / "dataset"
ASSETS_DIR = ROOT_DIR / "assets"
HDRI_DIR = ASSETS_DIR / "hdri"
WOOD_PATH = str(ASSETS_DIR / "1.webp")
SKY_PATH = str(ASSETS_DIR / "2.jpg")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_IMAGES = 10000
RENDER_SIZE = 256
SEED = 42
GLASS_HEIGHT = 2.0
GLASS_RADIUS_TOP = 0.55
GLASS_RADIUS_BOTTOM = 0.45
LIQUID_RADIUS = 0.42
ICE_CHANCE = 0.30  # 30% of frames get ice cubes

random.seed(SEED)

(OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "labels").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Utility: 3D → 2D YOLO bounding box
# ---------------------------------------------------------------------------


def corners_to_yolo(world_corners, scene, camera):
    """Project world-space corners to YOLO bbox.
    Blender camera y=0 at BOTTOM; YOLO y=0 at TOP — we flip Y."""
    coords_2d = []
    for wc in world_corners:
        co = world_to_camera_view(scene, camera, wc)
        coords_2d.append((co.x, 1.0 - co.y, co.z))  # flip Y for YOLO

    visible = [(x, y) for x, y, z in coords_2d if z > 0]
    if len(visible) < 2:
        return None

    xs = [p[0] for p in visible]
    ys = [p[1] for p in visible]
    x_min = max(0.0, min(1.0, min(xs)))
    x_max = max(0.0, min(1.0, max(xs)))
    y_min = max(0.0, min(1.0, min(ys)))
    y_max = max(0.0, min(1.0, max(ys)))

    w = x_max - x_min
    h = y_max - y_min
    if w <= 0.001 or h <= 0.001:
        return None

    return (x_min + w / 2, y_min + h / 2, w, h)


def compute_yolo_bbox(obj, scene, camera):
    """Project object's evaluated world-space bbox to YOLO coords."""
    local_corners = [Vector(corner) for corner in obj.bound_box]
    world_corners = [obj.matrix_world @ corner for corner in local_corners]
    return corners_to_yolo(world_corners, scene, camera)


def liquid_world_corners(liquid_height):
    """Explicit world-space corners for the liquid cylinder.
    Liquid is at x=0, y=0, from z=0 to z=liquid_height, radius=LIQUID_RADIUS."""
    r = LIQUID_RADIUS
    return [
        Vector((-r, -r, 0)),
        Vector((-r, -r, liquid_height)),
        Vector((-r, r, liquid_height)),
        Vector((-r, r, 0)),
        Vector((r, -r, 0)),
        Vector((r, -r, liquid_height)),
        Vector((r, r, liquid_height)),
        Vector((r, r, 0)),
    ]


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------


def make_glass_material(name="GlassMat"):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    glass = nodes.new(type="ShaderNodeBsdfGlass")
    glass.location = (0, 0)
    glass.inputs["Color"].default_value = (0.92, 0.95, 0.98, 1.0)
    glass.inputs["Roughness"].default_value = 0.03
    glass.inputs["IOR"].default_value = 1.45

    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (200, 0)
    links.new(glass.outputs["BSDF"], output.inputs["Surface"])
    return mat


def make_ice_material(name="IceMat"):
    """Frozen water — same IOR as water but clear and slightly cloudy."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Mix glass with a tiny bit of white volume for frostiness
    glass = nodes.new(type="ShaderNodeBsdfGlass")
    glass.location = (-200, 100)
    glass.inputs["Color"].default_value = (0.95, 0.97, 1.0, 1.0)
    glass.inputs["Roughness"].default_value = 0.04
    glass.inputs["IOR"].default_value = 1.31

    glossy = nodes.new(type="ShaderNodeBsdfGlossy")
    glossy.location = (-200, -100)
    glossy.inputs["Color"].default_value = (0.9, 0.92, 0.95, 1.0)
    glossy.inputs["Roughness"].default_value = 0.08

    mix = nodes.new(type="ShaderNodeMixShader")
    mix.location = (100, 0)
    mix.inputs["Fac"].default_value = 0.15
    links.new(glass.outputs["BSDF"], mix.inputs[1])
    links.new(glossy.outputs["BSDF"], mix.inputs[2])

    # Volume: very subtle absorption for depth
    vol_absorb = nodes.new(type="ShaderNodeVolumeAbsorption")
    vol_absorb.location = (100, -200)
    vol_absorb.inputs["Color"].default_value = (0.8, 0.85, 0.95, 1.0)
    vol_absorb.inputs["Density"].default_value = 0.10

    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (400, 0)
    links.new(mix.outputs["Shader"], output.inputs["Surface"])
    links.new(vol_absorb.outputs["Volume"], output.inputs["Volume"])
    return mat


def make_liquid_material(name="LiquidMat"):
    """Base liquid material — node tree will be rebuilt per frame."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    return mat


def randomize_liquid_material(mat):
    """Rebuild the liquid node tree for a random palette.
    Water (70%): Glass BSDF — physically correct transparent refraction.
    Juice (15%): Principled BSDF with transmission — colored semi-transparent.
    Coffee (10%): Principled BSDF with low transmission — dark, almost opaque.
    Milk (5%): Principled BSDF, no transmission — opaque white."""
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (400, 0)

    palette = random.random()

    if palette < 0.70:
        # ── WATER: Principled BSDF with max transmission ──
        # Glass BSDF inside another Glass BSDF causes dark nested-refraction
        # artifacts.  Principled BSDF with Transmission=1.0 is effectively
        # transparent and handles the glass/water boundary cleanly.
        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)
        bsdf.inputs["Base Color"].default_value = (
            random.uniform(0.97, 1.0),
            random.uniform(0.98, 1.0),
            random.uniform(0.99, 1.0),
            1.0,
        )
        bsdf.inputs["Roughness"].default_value = random.uniform(0.01, 0.03)
        bsdf.inputs["IOR"].default_value = random.uniform(1.32, 1.34)
        bsdf.inputs["Transmission Weight"].default_value = random.uniform(0.97, 1.0)
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
        return "water"

    elif palette < 0.85:
        # ── JUICE: Principled BSDF, colored, semi-transparent ──
        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)
        bsdf.inputs["Base Color"].default_value = (
            random.uniform(0.80, 0.95),
            random.uniform(0.30, 0.55),
            random.uniform(0.05, 0.15),
            1.0,
        )
        bsdf.inputs["Roughness"].default_value = random.uniform(0.08, 0.18)
        bsdf.inputs["IOR"].default_value = random.uniform(1.34, 1.36)
        bsdf.inputs["Transmission Weight"].default_value = random.uniform(0.30, 0.55)
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

        vol = nodes.new(type="ShaderNodeVolumeAbsorption")
        vol.location = (200, -150)
        vol.inputs["Color"].default_value = (
            random.uniform(0.8, 0.95),
            random.uniform(0.3, 0.5),
            random.uniform(0.02, 0.10),
            1.0,
        )
        vol.inputs["Density"].default_value = random.uniform(0.30, 0.60)
        links.new(vol.outputs["Volume"], output.inputs["Volume"])
        return "juice"

    elif palette < 0.95:
        # ── COFFEE: Principled BSDF, dark, almost opaque ──
        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)
        bsdf.inputs["Base Color"].default_value = (
            random.uniform(0.06, 0.15),
            random.uniform(0.03, 0.08),
            random.uniform(0.01, 0.04),
            1.0,
        )
        bsdf.inputs["Roughness"].default_value = random.uniform(0.10, 0.25)
        bsdf.inputs["IOR"].default_value = random.uniform(1.33, 1.35)
        bsdf.inputs["Transmission Weight"].default_value = random.uniform(0.05, 0.20)
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

        vol = nodes.new(type="ShaderNodeVolumeAbsorption")
        vol.location = (200, -150)
        vol.inputs["Color"].default_value = (
            random.uniform(0.05, 0.12),
            random.uniform(0.02, 0.06),
            random.uniform(0.01, 0.03),
            1.0,
        )
        vol.inputs["Density"].default_value = random.uniform(0.60, 1.20)
        links.new(vol.outputs["Volume"], output.inputs["Volume"])
        return "coffee"

    else:
        # ── MILK: Principled BSDF, opaque white, no transmission ──
        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)
        w = random.uniform(0.88, 0.97)
        bsdf.inputs["Base Color"].default_value = (
            w,
            w,
            w * random.uniform(0.96, 1.0),
            1.0,
        )
        bsdf.inputs["Roughness"].default_value = random.uniform(0.15, 0.35)
        bsdf.inputs["IOR"].default_value = random.uniform(1.34, 1.36)
        bsdf.inputs["Transmission Weight"].default_value = 0.0
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

        vol = nodes.new(type="ShaderNodeVolumeAbsorption")
        vol.location = (200, -150)
        vol.inputs["Color"].default_value = (0.92, 0.92, 0.90, 1.0)
        vol.inputs["Density"].default_value = random.uniform(0.80, 1.50)
        links.new(vol.outputs["Volume"], output.inputs["Volume"])
        return "milk"


def make_meniscus_material(name="MeniscusMat"):
    """Thin meniscus ring material — transparent like water but catches
    light at grazing angles via a Fresnel-driven Glass+Glossy mix."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Glass BSDF — same IOR as water, fully transparent
    glass = nodes.new(type="ShaderNodeBsdfGlass")
    glass.location = (-200, 100)
    glass.inputs["Color"].default_value = (0.98, 0.99, 1.0, 1.0)
    glass.inputs["Roughness"].default_value = 0.02
    glass.inputs["IOR"].default_value = 1.33

    # Glossy BSDF — catches specular highlights at the water line
    glossy = nodes.new(type="ShaderNodeBsdfGlossy")
    glossy.location = (-200, -100)
    glossy.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    glossy.inputs["Roughness"].default_value = 0.04

    # Layer Weight → Fresnel (more glossy at grazing angles, more glass head-on)
    layer = nodes.new(type="ShaderNodeLayerWeight")
    layer.location = (-500, 0)
    layer.inputs["Blend"].default_value = 0.25

    # Mix: glass (head-on) + glossy (grazing/edge)
    mix = nodes.new(type="ShaderNodeMixShader")
    mix.location = (100, 0)
    links.new(layer.outputs["Fresnel"], mix.inputs["Fac"])
    links.new(glass.outputs["BSDF"], mix.inputs[1])
    links.new(glossy.outputs["BSDF"], mix.inputs[2])

    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (400, 0)
    links.new(mix.outputs["Shader"], output.inputs["Surface"])
    return mat


def make_wood_floor(name="WoodFloor"):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    tex_coord.location = (-400, 0)
    mapping = nodes.new(type="ShaderNodeMapping")
    mapping.location = (-200, 0)
    mapping.inputs["Scale"].default_value = (2.0, 2.0, 1.0)
    links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])

    img_tex = nodes.new(type="ShaderNodeTexImage")
    img_tex.location = (0, 0)
    if os.path.exists(WOOD_PATH):
        img_tex.image = bpy.data.images.load(WOOD_PATH)
        print(f"  Loaded wood: {WOOD_PATH}")
    links.new(mapping.outputs["Vector"], img_tex.inputs["Vector"])

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (200, 0)
    bsdf.inputs["Roughness"].default_value = 0.5
    links.new(img_tex.outputs["Color"], bsdf.inputs["Base Color"])

    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (400, 0)
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat, mapping


# ---------------------------------------------------------------------------
# HDRI scanning & world setup
# ---------------------------------------------------------------------------


def scan_hdri_files():
    """Return list of .exr / .hdr paths in assets/hdri/."""
    if not HDRI_DIR.is_dir():
        return []
    hdris = sorted(
        [str(p) for p in HDRI_DIR.glob("*.exr")]
        + [str(p) for p in HDRI_DIR.glob("*.hdr")]
    )
    print(f"  Found {len(hdris)} HDRIs in {HDRI_DIR}")
    return hdris


def make_sky_world(hdri_path=None):
    """Set up world nodes.  If hdri_path given, load that environment texture."""
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    # Mapping node for HDRI rotation
    mapping = nodes.new(type="ShaderNodeMapping")
    mapping.location = (-600, 0)

    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    tex_coord.location = (-800, 0)
    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])

    env_tex = nodes.new(type="ShaderNodeTexEnvironment")
    env_tex.location = (-400, 0)
    if hdri_path and os.path.exists(hdri_path):
        env_tex.image = bpy.data.images.load(hdri_path)
        print(f"  Loaded HDRI: {Path(hdri_path).name}")
    elif os.path.exists(SKY_PATH):
        env_tex.image = bpy.data.images.load(SKY_PATH)
        print(f"  Loaded sky: {SKY_PATH}")
    links.new(mapping.outputs["Vector"], env_tex.inputs["Vector"])

    bg = nodes.new(type="ShaderNodeBackground")
    bg.location = (-100, 0)
    bg.inputs["Strength"].default_value = 1.0
    links.new(env_tex.outputs["Color"], bg.inputs["Color"])

    output = nodes.new(type="ShaderNodeOutputWorld")
    output.location = (100, 0)
    links.new(bg.outputs["Background"], output.inputs["Surface"])

    return world, env_tex, mapping


# ---------------------------------------------------------------------------
# Ice cube helpers
# ---------------------------------------------------------------------------


def spawn_ice_cubes(liquid_height, ice_mat):
    """Create 1-3 ice cubes floating at the water surface like icebergs.
    Each cube straddles the water line: ~40% above, ~60% below."""
    count = random.randint(1, 3)
    cubes = []
    for _ in range(count):
        size = random.uniform(0.08, 0.18)
        # Position at the surface: random XY within liquid, Z straddles the water line
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(0.05, LIQUID_RADIUS - size * 0.6)
        px = radius * math.cos(angle)
        py = radius * math.sin(angle)
        # Cube center sits BELOW the water line — ~30% above (tip), ~70% below (mass)
        pz = liquid_height - size * random.uniform(0.15, 0.35)

        bpy.ops.mesh.primitive_cube_add(size=size, location=(px, py, pz))
        cube = bpy.context.active_object
        cube.name = f"IceCube_{len(cubes)}"
        cube.data.materials.append(ice_mat)

        # Random rotation — ice chunks don't sit perfectly flat
        cube.rotation_euler = (
            random.uniform(-0.3, 0.3),
            random.uniform(-0.3, 0.3),
            random.uniform(0, math.pi * 2),
        )
        cubes.append(cube)
        print(f"    Iceberg at ({px:.2f}, {py:.2f}, z={pz:.2f}) size={size:.2f}")

    return cubes


def cleanup_ice_cubes():
    """Remove all ice cube objects from the scene."""
    for obj in bpy.data.objects:
        if obj.name.startswith("IceCube_"):
            bpy.data.objects.remove(obj, do_unlink=True)


# ---------------------------------------------------------------------------
# Compositor: grain overlay
# ---------------------------------------------------------------------------


def setup_compositor():
    """Add a subtle noise grain overlay to the compositor output.
    Returns the Mix node (for per-frame Fac tweaking) or None if unavailable."""
    scene = bpy.context.scene
    try:
        scene.use_nodes = True
    except Exception:
        print("  Compositor unavailable (headless mode) — skipping grain")
        return None

    # In Blender 5.x the scene node_tree may be a different API shape;
    # use getattr for compatibility.
    tree = getattr(scene, "node_tree", None)
    if tree is None:
        print("  Compositor node_tree not available — skipping grain")
        return None

    nodes = tree.nodes
    links = tree.links
    nodes.clear()

    rl = nodes.new(type="CompositorNodeRLayers")
    rl.location = (0, 0)

    # Noise texture for grain
    noise = nodes.new(type="CompositorNodeTexNoise")
    noise.location = (200, -150)

    # Scale noise to full image
    scale_node = nodes.new(type="CompositorNodeScale")
    scale_node.location = (200, -300)
    scale_node.space = "RENDER_SIZE"
    links.new(noise.outputs["Color"], scale_node.inputs["Image"])

    # Mix noise over the render (low opacity)
    mix = nodes.new(type="CompositorNodeMixRGB")
    mix.location = (400, 0)
    mix.blend_type = "OVERLAY"
    mix.inputs["Fac"].default_value = 0.0  # will be randomized per frame
    links.new(rl.outputs["Image"], mix.inputs[1])
    links.new(scale_node.outputs["Image"], mix.inputs[2])

    output = nodes.new(type="CompositorNodeComposite")
    output.location = (600, 0)
    links.new(mix.outputs["Image"], output.inputs["Image"])

    return mix  # so we can tweak the Fac per frame


# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------


def setup_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    scene = bpy.context.scene

    # --- Cycles render settings ---
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = RENDER_SIZE
    scene.render.resolution_y = RENDER_SIZE
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False

    # GPU compute (graceful fallback to CPU if CUDA unavailable)
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        for dev in prefs.devices:
            dev.use = True
        scene.cycles.device = "GPU"
        print(f"  Cycles: GPU (CUDA)")
    except Exception:
        scene.cycles.device = "CPU"
        print(f"  Cycles: CPU fallback")

    # Speed/quality balance for 256×256
    # Nested glass+water needs enough bounces: ray enters outer glass,
    # passes inner glass wall, enters water, exits water, exits glass.
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.05
    scene.cycles.max_bounces = 12  # must be >= transmission_bounces
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.cycles.transmission_bounces = 12  # glass + water nested refraction
    scene.cycles.transparent_max_bounces = 12  # transparent shadow rays
    scene.cycles.volume_bounces = 2  # ice + liquid volume

    print(f"  Cycles: {scene.cycles.samples} samples, denoised")

    # --- Materials ---
    glass_mat = make_glass_material()
    liquid_mat = make_liquid_material()
    meniscus_mat = make_meniscus_material()
    ice_mat = make_ice_material()
    wood_mat, wood_mapping = make_wood_floor()
    world, sky_tex, hdri_mapping = make_sky_world()

    # --- Glass mesh (tapered cylinder) ---
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=48,
        radius=GLASS_RADIUS_TOP,
        depth=GLASS_HEIGHT,
        location=(0, 0, GLASS_HEIGHT / 2),
    )
    glass_obj = bpy.context.active_object
    glass_obj.name = "Glass"
    glass_obj.data.materials.append(glass_mat)

    for vert in glass_obj.data.vertices:
        if vert.co.z < 0.01:
            factor = GLASS_RADIUS_BOTTOM / GLASS_RADIUS_TOP
            vert.co.x *= factor
            vert.co.y *= factor

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    solidify = glass_obj.modifiers.new(name="Solidify", type="SOLIDIFY")
    solidify.thickness = 0.025
    solidify.offset = 0

    # --- Liquid mesh ---
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=48,
        radius=LIQUID_RADIUS,
        depth=1.0,
        location=(0, 0, 0.5),
    )
    liquid_obj = bpy.context.active_object
    liquid_obj.name = "Liquid"
    liquid_obj.data.materials.append(liquid_mat)

    # -- Overlap removed — Principled BSDF with Transmission handles nesting cleanly

    # -- Meniscus: thin torus ring at the water-glass boundary --
    # Catches light differently than the water body via its own material.
    bpy.ops.mesh.primitive_torus_add(
        major_radius=LIQUID_RADIUS * 0.97,
        minor_radius=0.008,
        location=(0, 0, 0.01),
    )
    meniscus_obj = bpy.context.active_object
    meniscus_obj.name = "Meniscus"
    meniscus_obj.data.materials.append(meniscus_mat)

    # --- Ground ---
    bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 0, -0.01))
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground.data.materials.append(wood_mat)

    # --- Camera ---
    bpy.ops.object.camera_add(location=(4.0, -2.0, 2.0))
    camera = bpy.context.active_object
    camera.name = "Camera"
    scene.camera = camera

    # --- Depth of Field ---
    camera.data.dof.use_dof = True
    camera.data.dof.focus_object = glass_obj
    camera.data.dof.aperture_fstop = 4.0  # will be randomized per frame

    # --- Lighting ---
    bpy.ops.object.light_add(type="SUN", location=(3, -4, 5))
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.data.energy = 4.0  # softer — glass shouldn't cast harsh shadows
    sun.data.angle = math.radians(12)  # wider = softer transparent shadows

    bpy.ops.object.light_add(type="AREA", location=(-1, 3, 1.5))
    fill = bpy.context.active_object
    fill.name = "Fill"
    fill.data.energy = 60.0
    fill.data.size = 3.0

    # --- Compositor ---
    grain_mix = setup_compositor()

    return (
        scene,
        camera,
        glass_obj,
        liquid_obj,
        meniscus_obj,
        sun,
        fill,
        ground,
        wood_mapping,
        sky_tex,
        hdri_mapping,
        ice_mat,
        grain_mix,
    )


# ---------------------------------------------------------------------------
# Per-frame randomization
# ---------------------------------------------------------------------------


def randomize_and_render(
    frame_idx,
    scene,
    camera,
    glass_obj,
    liquid_obj,
    meniscus_obj,
    sun,
    fill,
    ground,
    wood_mapping,
    sky_tex,
    hdri_mapping,
    ice_mat,
    grain_mix,
    hdri_paths,
):
    # ── 1. Liquid height ──
    liquid_fill_ratio = random.uniform(0.02, 0.98)
    liquid_height = liquid_fill_ratio * GLASS_HEIGHT
    liquid_obj.scale.z = liquid_height
    liquid_obj.location.z = liquid_height / 2

    # ── 2. Meniscus ring follows liquid height ──
    meniscus_obj.location.z = liquid_height

    # ── 3. Liquid color palette ──
    liquid_mat = liquid_obj.data.materials[0]
    palette = randomize_liquid_material(liquid_mat)

    # ── 4. Glass tint ──
    glass_mat = glass_obj.data.materials[0]
    if glass_mat.node_tree:
        for node in glass_mat.node_tree.nodes:
            if node.type == "BSDF_GLASS":
                tr = random.uniform(0.85, 0.96)
                tg = random.uniform(0.90, 0.98)
                tb = random.uniform(0.93, 1.0)
                node.inputs["Color"].default_value = (tr, tg, tb, 1.0)
                break

    # ── 5. Camera orbit ──
    azimuth = random.uniform(0, 2 * math.pi)
    elevation = random.uniform(math.radians(15), math.radians(55))
    distance = random.uniform(4.0, 7.0)

    cam_x = distance * math.cos(elevation) * math.cos(azimuth)
    cam_y = distance * math.cos(elevation) * math.sin(azimuth)
    cam_z = distance * math.sin(elevation)
    camera.location = Vector((cam_x, cam_y, cam_z))

    target = Vector(
        (
            random.uniform(-0.10, 0.10),
            random.uniform(-0.10, 0.10),
            GLASS_HEIGHT / 2 + random.uniform(-0.10, 0.10),
        )
    )
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    # ── 6. Depth of Field ──
    camera.data.dof.aperture_fstop = random.uniform(1.4, 8.0)

    # ── 7. Lighting ──
    sun.data.energy = random.uniform(3.0, 8.0)
    sun.location = (random.uniform(1, 5), random.uniform(-5, -1), random.uniform(3, 6))
    fill.data.energy = random.uniform(30, 90)

    # ── 8. Wood texture variation ──
    wood_mapping.inputs["Scale"].default_value = (
        random.uniform(1.5, 3.5),
        random.uniform(1.5, 3.5),
        1.0,
    )
    wood_mapping.inputs["Rotation"].default_value[2] = random.uniform(0, math.pi)

    # ── 9. HDRI: pick random file, rotate mapping Z ──
    if hdri_paths:
        chosen = random.choice(hdri_paths)
        try:
            world_nodes = bpy.context.scene.world.node_tree.nodes
            for node in world_nodes:
                if node.type == "TEX_ENVIRONMENT":
                    if os.path.exists(chosen):
                        node.image = bpy.data.images.load(chosen)
                    break
        except Exception:
            pass
        # Rotate HDRI for different shadow/specular directions
        try:
            hdri_mapping.inputs["Rotation"].default_value[2] = random.uniform(
                0, math.pi * 2
            )
        except Exception:
            pass

    # ── 10. Ice cubes (30% chance) ──
    cleanup_ice_cubes()
    if random.random() < ICE_CHANCE:
        spawn_ice_cubes(liquid_height, ice_mat)

    # ── 11. Grain strength ──
    if grain_mix is not None:
        grain_mix.inputs["Fac"].default_value = random.uniform(0.03, 0.12)

    # ── Render ──
    filepath = str(OUTPUT_DIR / "images" / f"glass_{frame_idx:05d}.png")
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)

    # ── YOLO labels ──
    depsgraph = bpy.context.evaluated_depsgraph_get()
    glass_eval = glass_obj.evaluated_get(depsgraph)

    glass_bbox = compute_yolo_bbox(glass_eval, scene, camera)
    liquid_bbox = corners_to_yolo(liquid_world_corners(liquid_height), scene, camera)

    if glass_bbox and liquid_bbox:
        bbox_ratio = liquid_bbox[3] / glass_bbox[3]
    else:
        bbox_ratio = -1

    label_path = str(OUTPUT_DIR / "labels" / f"glass_{frame_idx:05d}.txt")
    with open(label_path, "w") as f:
        if glass_bbox is not None:
            f.write(
                f"0 {glass_bbox[0]:.6f} {glass_bbox[1]:.6f} "
                f"{glass_bbox[2]:.6f} {glass_bbox[3]:.6f}\n"
            )
        if liquid_bbox is not None:
            f.write(
                f"1 {liquid_bbox[0]:.6f} {liquid_bbox[1]:.6f} "
                f"{liquid_bbox[2]:.6f} {liquid_bbox[3]:.6f}\n"
            )

    return liquid_fill_ratio, bbox_ratio, palette


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    args = {"num_images": NUM_IMAGES, "start_index": 0}
    argv = sys.argv
    if "--" in argv:
        extra = argv[argv.index("--") + 1 :]
        i = 0
        while i < len(extra):
            if extra[i] == "--num-images" and i + 1 < len(extra):
                args["num_images"] = int(extra[i + 1])
                i += 2
            elif extra[i] == "--start-index" and i + 1 < len(extra):
                args["start_index"] = int(extra[i + 1])
                i += 2
            else:
                i += 1
    return args


def main():
    args = parse_args()
    num_images = args["num_images"]
    start_index = args["start_index"]

    print(
        f"[GlassGen] Cycles GPU | {num_images} images (start={start_index}) | {RENDER_SIZE}x{RENDER_SIZE}"
    )

    # Scan HDRIs before setup (they'll be loaded per-frame)
    hdri_paths = scan_hdri_files()

    print(f"[GlassGen] Setting up scene...")
    (
        scene,
        camera,
        glass_obj,
        liquid_obj,
        meniscus_obj,
        sun,
        fill,
        ground,
        wood_mapping,
        sky_tex,
        hdri_mapping,
        ice_mat,
        grain_mix,
    ) = setup_scene()

    print(f"[GlassGen] Rendering...")
    for i in range(num_images):
        idx = start_index + i
        fill_ratio, bbox_ratio, palette = randomize_and_render(
            idx,
            scene,
            camera,
            glass_obj,
            liquid_obj,
            meniscus_obj,
            sun,
            fill,
            ground,
            wood_mapping,
            sky_tex,
            hdri_mapping,
            ice_mat,
            grain_mix,
            hdri_paths,
        )
        if i % 50 == 0:
            print(
                f"  [{idx}/{start_index + num_images}] fill={fill_ratio:.2f} "
                f"bbox_ratio={bbox_ratio:.3f}  liquid={palette}"
            )

    # Final cleanup
    cleanup_ice_cubes()

    meta = {
        "num_images": num_images,
        "render_size": RENDER_SIZE,
        "classes": ["glass", "liquid"],
        "glass_height": GLASS_HEIGHT,
        "seed": SEED,
        "renderer": "Cycles",
        "domain_randomization": {
            "liquid_palettes": ["water", "juice", "coffee", "milk"],
            "ice_cubes_chance": ICE_CHANCE,
            "hdri_count": len(hdri_paths),
            "dof_fstop_range": [1.4, 8.0],
            "grain_range": [0.03, 0.12],
        },
    }
    with open(str(OUTPUT_DIR / "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[GlassGen] Done! → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
