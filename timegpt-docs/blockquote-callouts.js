/*
 * Blockquote callouts: style nbdev/Quarto callouts like Mintlify's.
 *
 * The library docs are exported from notebooks, so a Quarto callout arrives as
 * a plain blockquote whose first paragraph is only a bold title
 * (`> **Warning**`). CSS can't read that text, so we tag the blockquote with
 * the matching variant and mark the title paragraph as its label. Free-form
 * titles ("The model", "Prerequisites") are left alone. Re-runs on client-side
 * nav and when accordions or tabs mount content late.
 */
(function () {
  var VARIANTS = {
    note: "note",
    tip: "tip",
    important: "info",
    "important note": "info",
    warning: "warning",
    caution: "warning",
    danger: "danger",
  };

  function tag(quote) {
    // Mintlify renders markdown paragraphs as <span data-as="p">.
    var first = quote.firstElementChild;
    if (!first || !first.matches('p, [data-as="p"]')) return;

    var strong = first.firstElementChild;
    if (!strong || strong.tagName !== "STRONG") return;
    if (first.textContent.trim() !== strong.textContent.trim()) return;

    var variant = VARIANTS[strong.textContent.trim().toLowerCase()];
    if (!variant) return;

    if (quote.getAttribute("data-blockquote-variant") !== variant) {
      quote.setAttribute("data-blockquote-variant", variant);
    }
    if (!first.hasAttribute("data-blockquote-label")) {
      first.setAttribute("data-blockquote-label", "");
    }
  }

  function run() {
    document.querySelectorAll(".prose blockquote").forEach(tag);
  }

  var scheduled = false;
  function schedule() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(function () {
      scheduled = false;
      run();
    });
  }

  if (document.readyState !== "loading") run();
  else document.addEventListener("DOMContentLoaded", run);

  new MutationObserver(schedule).observe(document.body, {
    childList: true,
    subtree: true,
  });
})();
