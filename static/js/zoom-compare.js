document.addEventListener('DOMContentLoaded', function () {
  // Enable hover zoom only on devices with real mouse + hover
  const isDesktop = window.matchMedia('(hover: hover) and (pointer: fine)').matches;

  /* =========================
     1) Zoom for #zoom-compare
     ========================= */
  (function setupZoomCompare() {
    const containers = document.querySelectorAll('#zoom-compare .zoom-container');
    if (!containers.length) return;

    const scale = 2.0; // zoom factor for teaser comparison

    const images = Array.from(containers)
      .map(c => c.querySelector('.zoom-sync'))
      .filter(Boolean);

    // If we are NOT on desktop, do not attach hover zoom here
    // (mobile can still use modal/tap behaviour if you added it).
    if (!isDesktop) {
      // Make sure they are at normal scale
      images.forEach(img => {
        img.style.transform = 'scale(1)';
        img.style.transformOrigin = '50% 50%';
      });
      return;
    }

    function setZoom(enabled) {
      images.forEach(img => {
        img.style.transform = enabled ? `scale(${scale})` : 'scale(1)';
      });
    }

    function setOrigin(rx, ry) {
      const ox = rx * 100;
      const oy = ry * 100;
      images.forEach(img => {
        img.style.transformOrigin = `${ox}% ${oy}%`;
      });
    }

    containers.forEach(container => {
      container.addEventListener('mouseenter', () => setZoom(true));
      container.addEventListener('mouseleave', () => setZoom(false));

      container.addEventListener('mousemove', function (e) {
        const rect = container.getBoundingClientRect();
        let rx = (e.clientX - rect.left) / rect.width;
        let ry = (e.clientY - rect.top) / rect.height;
        rx = Math.min(Math.max(rx, 0), 1);
        ry = Math.min(Math.max(ry, 0), 1);
        setOrigin(rx, ry);
      });
    });
  })();

  /* =======================================
     2) Zoom for "Comparing methods" grid
     ======================================= */
  (function setupComparisonGridZoom() {
    const containers = document.querySelectorAll('.comparison-grid .zoom-container');
    if (!containers.length) return;

    const scale = 3.0; // stronger zoom for method comparison

    // On non-desktop (phones/tablets), do not attach hover zoom
    if (!isDesktop) {
      containers.forEach(container => {
        const img = container.querySelector('.zoom-sync');
        if (img) {
          img.style.transform = 'scale(1)';
          img.style.transformOrigin = '50% 50%';
        }
      });
      return;
    }

    containers.forEach(container => {
      const img = container.querySelector('.zoom-sync');
      if (!img) return;

      container.addEventListener('mouseenter', function () {
        img.style.transform = `scale(${scale})`;
      });

      container.addEventListener('mouseleave', function () {
        img.style.transform = 'scale(1)';
        img.style.transformOrigin = '50% 50%';
      });

      container.addEventListener('mousemove', function (e) {
        const rect = container.getBoundingClientRect();
        let rx = (e.clientX - rect.left) / rect.width;
        let ry = (e.clientY - rect.top) / rect.height;
        rx = Math.min(Math.max(rx, 0), 1);
        ry = Math.min(Math.max(ry, 0), 1);

        const ox = rx * 100;
        const oy = ry * 100;
        img.style.transformOrigin = `${ox}% ${oy}%`;
      });
    });
  })();
});



document.addEventListener('DOMContentLoaded', function () {
    const modal = document.getElementById('comparison-modal');
    const modalImg = document.getElementById('comparison-modal-image');
    const modalCaption = document.getElementById('comparison-modal-caption');
    const closeBtn = document.querySelector('.comparison-modal-close');
    const backdrop = document.querySelector('.comparison-modal-backdrop');

    // Decide if we want "tap shows modal" behaviour
    function isSmallOrTouch() {
      return window.matchMedia('(max-width: 768px)').matches ||
             window.matchMedia('(hover: none)').matches;
    }

    function attachHandlers() {
      const shouldUseModal = isSmallOrTouch();

      document.querySelectorAll('.comparison-cell img').forEach(function (img) {
        // Remove old listeners if any
        img.onclick = null;

        if (shouldUseModal) {
          img.onclick = function () {
            modalImg.src = img.src;
            modalImg.alt = img.alt || '';
            modalCaption.textContent = img.alt || '';
            modal.classList.add('is-active');
          };
        }
        // On desktop, no click handler → only hover zoom
      });
    }

    function closeModal() {
      modal.classList.remove('is-active');
      modalImg.src = '';
      modalCaption.textContent = '';
    }

    closeBtn.addEventListener('click', closeModal);
    backdrop.addEventListener('click', closeModal);

    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && modal.classList.contains('is-active')) {
        closeModal();
      }
    });

    // Initial attach
    attachHandlers();

    // Re-evaluate on resize/orientation change (optional, but nice)
    window.addEventListener('resize', function () {
      attachHandlers();
    });
  });