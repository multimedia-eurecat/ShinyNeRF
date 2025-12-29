
document.addEventListener('DOMContentLoaded', function () {
  // Discrete physical values
  const kappaValues  = [10, 16, 25, 40, 63, 100, 160, 200];                // Nk = 8
  const eValues      = [0.00, 0.25, 0.50, 0.75, 1.00];    // Ne = 5
  const angleDegVals = [0, 36, 72, 108, 144];      // Nθ = 5

  // DOM elements
  const kappaSlider = document.getElementById('kappa-slider');
  const eSlider     = document.getElementById('e-slider');
  const angleSlider = document.getElementById('angle-slider');

  const kappaLabel  = document.getElementById('kappa-value');
  const eLabel      = document.getElementById('e-value');
  const angleLabel  = document.getElementById('angle-value');

  const imgEl       = document.getElementById('param-image');
  const captionEl   = document.getElementById('param-caption');

  if (!kappaSlider || !eSlider || !angleSlider || !imgEl) {
    return;
  }

  // Initial slider positions (indices)
  kappaSlider.value = 0;
  eSlider.value     = 0;
  angleSlider.value = 0;

  function updateParamImage() {
    const ki = parseInt(kappaSlider.value, 10);
    const ei = parseInt(eSlider.value, 10);
    const ai = parseInt(angleSlider.value, 10);

    const kappa    = kappaValues[ki];    // e.g. 10, 16, ...
    const e        = eValues[ei];        // e.g. 0.00, 0.25, ...
    const angleDeg = angleDegVals[ai];   // e.g. 0, 36, ...
    kappaLabel.textContent = kappa.toString();
    eLabel.textContent     = e.toFixed(2);
    angleLabel.textContent = angleDeg + '°';

    // IMPORTANT: filenames must match this exact pattern:
    //   image<kappa>-<e with 2 decimals>-<angleDeg>.png
    // Example: image10-1.00-0.png
    const eStr = e.toFixed(2); // "0.00", "0.25", "1.00"
    const src  = `images/slider_images/image${kappa}-${eStr}-${angleDeg}.png`;

    imgEl.src = src;

    if (captionEl) {
        captionEl.innerHTML =
        `&kappa; = ${kappa}, ` +
        `e = ${eStr}, ` +
        `&phi; = ${angleDeg}&deg;`;
    }
    }

  // Listen to slider changes (continuous updates while dragging)
  ['input', 'change'].forEach(evt => {
    kappaSlider.addEventListener(evt, updateParamImage);
    eSlider.addEventListener(evt, updateParamImage);
    angleSlider.addEventListener(evt, updateParamImage);
  });

  // Initial update
  updateParamImage();
});

