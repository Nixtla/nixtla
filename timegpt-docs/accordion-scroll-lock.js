/*
 * Accordion open/close: keep the scroll position steady.
 *
 * Mintlify deep-links accordions — opening one sets location.hash to the
 * accordion id, which makes the browser scroll to that anchor (scroll-margin-top
 * + html { scroll-behavior: smooth }). That reads as the page "jumping". CSS
 * can't cancel fragment navigation, so we pin the scroll position for a short
 * window after a summary click, defeating the anchor scroll while KEEPING the
 * hash (the shareable link still works). Runs in dev and prod.
 */
(function () {
  var SEL = "details.accordion > summary, details.expandable > summary";
  var PIN_MS = 700; // covers the 0.4s open animation + smooth-scroll frames

  document.addEventListener(
    "click",
    function (e) {
      var summary = e.target.closest && e.target.closest(SEL);
      if (!summary) return;

      var y = window.scrollY;
      var start = -1;

      function pin(now) {
        if (start < 0) start = now;
        // Only correct if something tried to move us — avoids fighting the user.
        if (window.scrollY !== y) window.scrollTo(0, y);
        if (now - start < PIN_MS) requestAnimationFrame(pin);
      }
      requestAnimationFrame(pin);
    },
    true
  );
})();
