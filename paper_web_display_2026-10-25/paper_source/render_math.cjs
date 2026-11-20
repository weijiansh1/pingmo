// Render locally; no browser-side math script or network is required.
const fs = require('node:fs');
const katex = require('../webpage/vendor/katex/katex.min.js');
const expressions = JSON.parse(fs.readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(expressions.map(({tex, display}) => katex.renderToString(tex, {
  displayMode: display,
  output: 'htmlAndMathml',
  throwOnError: true,
  strict: 'error',
  trust: false,
}))));
