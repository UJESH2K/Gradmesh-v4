# 3D models

## `training-rig.glb`

Drop a GLB here with exactly this filename and the training screen picks it up
on the next page load. No code change, no rebuild.

Until a file exists, `components/dashboard/TrainingRig.tsx` renders a procedural
GPU fan instead, so the screen works either way.

### What the file should contain

- **One object, authored around the origin.** It gets auto-scaled to about two
  units and re-centred on load, so absolute size does not matter, but a model
  built far from the origin will look off-centre.
- **Baked animation clips that loop cleanly.** Every clip in the file is played
  at once, so a robot arm with one clip per joint works with no configuration.
  The first frame and the last frame should match, or the loop will visibly pop.
- **Embedded textures.** External texture files are not resolved.
- **Under about 5 MB.** It is loaded by every dashboard visitor.

### How it behaves

Playback speed is driven by how much of the mesh is busy, so the rig runs faster
under load and idles slowly when nothing is training. Its accent light changes
colour with the run state: blue while training, cyan while aggregating weights,
green when finished, red on failure.

A model with no animation clips is slowly rotated instead.
