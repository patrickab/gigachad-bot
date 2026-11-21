/** Turbopack loader: an svg imported by a Jupyter package's script is the markup string JupyterLab's LabIcon expects, see next.config.mjs. */
module.exports = function svgMarkupLoader(source) {
  return `export default ${JSON.stringify(source)}`
}
