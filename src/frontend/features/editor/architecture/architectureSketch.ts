import rough from "roughjs"

// Rough.js is the same seeded, multi-stroke renderer Excalidraw uses. Stable
// seeds keep the drawing still across React renders instead of making it jitter.
const roughGenerator = rough.generator()

/** A stable non-zero seed for an id, so its sketch never changes between renders. */
export function roughSeed(id: string): number {
  let hash = 2166136261
  for (let index = 0; index < id.length; index += 1) hash = Math.imul(hash ^ id.charCodeAt(index), 16777619)
  return (hash >>> 0) || 1
}

// Curved connections draw as a single wobbly line: rough.js's doubled stroke
// reads as jitter at edge scale rather than pencil. Elbow connections stay crisp.
export function edgeSketchPaths(d: string, seed: number, strokeWidth: number) {
  return roughGenerator.toPaths(roughGenerator.path(d, {
    seed, roughness: 1, bowing: 1, stroke: "currentColor", strokeWidth, preserveVertices: true, disableMultiStroke: true,
  }))
}

// A rounded rectangle, sketched by rough.js like Excalidraw's boxes. The fill
// is sketched too, so it follows the outline's wobble instead of sitting under
// it as a perfect CSS shape. rough.js only echoes `fill` back on the fill path;
// the actual colour comes from CSS so it tracks the theme.
export function nodeSketchPaths(width: number, height: number, seed: number, strokeWidth: number) {
  const options = { seed, roughness: 1, bowing: 1, stroke: "currentColor", strokeWidth, preserveVertices: true, fill: "sketch", fillStyle: "solid" }
  const radius = Math.max(0, Math.min(9, (width - 2) / 2, (height - 2) / 2))
  const d = `M ${radius} 1 H ${width - radius} Q ${width - 1} 1 ${width - 1} ${radius} V ${height - radius} Q ${width - 1} ${height - 1} ${width - radius} ${height - 1} H ${radius} Q 1 ${height - 1} 1 ${height - radius} V ${radius} Q 1 1 ${radius} 1 Z`
  return roughGenerator.toPaths(roughGenerator.path(d, options))
}
