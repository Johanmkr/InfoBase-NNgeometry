window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    macros: {
      R: "\\mathbb{R}",
      N: "\\mathbb{N}",
      C: "\\mathbb{C}",
      Z: "\\mathbb{Z}",
      Q: "\\mathbb{Q}",
      E: "\\mathbb{E}",
      norm: ["\\left\\lVert #1 \\right\\rVert", 1]
    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  if (!window.MathJax || !MathJax.startup || !MathJax.startup.output) return;  // CDN not loaded (offline)
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
