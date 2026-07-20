/*
 * Code-group tabs: show the block's label next to the icon.
 *
 * Mintlify renders each code-group tab as an icon only; the human label lives in
 * the tab's panel header ([data-component-part="code-block-header-filename"]).
 * CSS can't copy text across elements, so we mirror each panel's filename into
 * its tab (next to the icon), falling back to "Example code" when the block has
 * no title. Re-runs on client-side nav and when accordions reveal code groups.
 */
(function () {
  var FALLBACK = "Example code";

  function labelTab(tab) {
    var titleBox = tab.querySelector('[class*="peer/title"]');
    if (!titleBox) return;

    var panel = document.getElementById(tab.getAttribute("aria-controls"));
    var fn =
      panel &&
      panel.querySelector(
        '[data-component-part="code-block-header-filename"] span'
      );
    var text = (fn && fn.textContent.trim()) || FALLBACK;

    var label = titleBox.querySelector("[data-cg-label]");
    if (!label) {
      label = document.createElement("span");
      label.setAttribute("data-cg-label", "");
      titleBox.appendChild(label);
    }
    if (label.textContent !== text) label.textContent = text;
  }

  function run() {
    document
      .querySelectorAll('.code-group [role="tab"]')
      .forEach(labelTab);
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

  // Re-run when tabs/panels mount late (hydration, client-side nav, accordions).
  new MutationObserver(schedule).observe(document.body, {
    childList: true,
    subtree: true,
  });
})();
