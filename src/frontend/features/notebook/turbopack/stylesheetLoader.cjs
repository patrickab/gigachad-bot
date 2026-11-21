/**
 * Turbopack loader: a stylesheet a Jupyter package imports as text (`?raw`) loads as a stylesheet of the page, which is what
 * webpack makes of that import, see next.config.mjs.
 */
module.exports = function stylesheetLoader() {
  const fileName = this.resourcePath.split(/[\\/]/).pop()
  return `import ${JSON.stringify(`./${fileName}`)}\nexport default ""`
}
