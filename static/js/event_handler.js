document.addEventListener('DOMContentLoaded', domReady);

        function domReady() {
            new Dics({
                container: document.querySelectorAll('.b-dics')[0],
                hideTexts: false,
                textPosition: "bottom"

            });
            new Dics({
                container: document.querySelectorAll('.b-dics')[1],
                hideTexts: false,
                textPosition: "bottom"

            });
        }

        function largeSceneEvent(idx) {
            let dics = document.querySelectorAll('.b-dics')[0]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 3
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                switch (idx) {
                    case 0:
                        image.src = 'assets/img/nvidia/';
                        break;
                    case 1:
                        image.src = 'assets/img/jhu/';
                        break;
                    case 2:
                        image.src = 'assets/img/Barn/';
                        break;
                    case 3:
                        image.src = 'assets/img/Caterpillar/';
                        break;
                    case 4:
                        image.src = 'assets/img/Courthouse/';
                        break;
                    case 5:
                        image.src = 'assets/img/Ignatius/';
                        break;
                    case 6:
                        image.src = 'assets/img/Meetingroom/';
                        break;
                    case 7:
                        image.src = 'assets/img/Truck/';
                        break;
                }
                switch (i) {
                    case 0:
                        image.src = image.src + '/rgb.png';
                        break;
                    case 1:
                        image.src = image.src + '/mesh.png';
                        break;
                    case 2:
                        image.src = image.src + '/normal.png';
                        break;
                }
            }

            scene_list = document.getElementById("large-scale-recon-1").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
            scene_list = document.getElementById("large-scale-recon-2").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i+2) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }

        function objectSceneEvent(idx) {
            let dics = document.querySelectorAll('.b-dics')[0]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 4
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                switch (idx) {
                    case 0:
                        image.src = 'resources/360_stmt_supp/bicycle_0';
                        break;
                    case 1:
                        image.src = 'resources/360_stmt_supp/bonsai_12';
                        break;
                    case 2:
                        image.src = 'resources/360_stmt_supp/counter_19';
                        break;
                    case 3:
                        image.src = 'resources/360_stmt_supp/flowers_8';
                        break;
                    case 4:
                        image.src = 'resources/360_stmt_supp/garden_1';
                        break;
                    case 5:
                        image.src = 'resources/360_stmt_supp/kitchen_0';
                        break;
                    case 6:
                        image.src = 'resources/360_stmt_supp/stump_0';
                        break;
                    case 7:
                        image.src = 'resources/360_stmt_supp/treehill_0';
                        break;    
                }
                switch (i) {
                    case 0:
                        image.src = image.src + '_ewa.jpg';
                        break;
                    case 1:
                        image.src = image.src + '_ours.jpg';
                        break;
                    case 2:
                        image.src = image.src + '_upgt.jpg';
                        break;
                    case 3:
                        image.src = image.src + '_gt.jpg';
                        break;

                }
            }

            let scene_list = document.getElementById("object-scale-recon").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }

        function ablation3DEvent(idx) {
            let dics = document.querySelectorAll('.b-dics')[1]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 4
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                switch (idx) {
                    case 0:
                        image.src = 'resources/360_stmt_ablation/bicycle_0';
                        break;
                    case 1:
                        image.src = 'resources/360_stmt_ablation/bicycle_3';
                        break;
                    case 2:
                        image.src = 'resources/360_stmt_ablation/bicycle_5';
                        break;
                    case 3:
                        image.src = 'resources/360_stmt_ablation/garden_0';
                        break; 
                    case 4:
                        image.src = 'resources/360_stmt_ablation/garden_1';
                        break;
                    case 5:
                        image.src = 'resources/360_stmt_ablation/treehill_9';
                        break; 
                }
                switch (i) {
                    case 0:
                        image.src = image.src + '_no3d.jpg';
                        break;
                    case 1:
                        image.src = image.src + '_ours.jpg';
                        break;
                    case 2:
                        image.src = image.src + '_upgt.jpg';
                        break;
                    case 3:
                        image.src = image.src + '_gt.jpg';
                        break;
                }
            }

            let scene_list = document.getElementById("ablation-3d-filter").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }

const LOOP_START = 0.001;
const WRAP_EPS = 0.12;     // wrap 120ms before end (avoid ended)
const HOLD_MS = 50;       // 0.1s hold

function seekSafe(v, t) {
  try {
    if (typeof v.fastSeek === "function") v.fastSeek(t);
    else v.currentTime = t;
  } catch {}
}

function makeGroupLooper(videos) {
  const master = videos[0];
  let loopDur = 0;
  let started = false;
  let wrapping = false;

  function wrapWithHold() {
    if (wrapping) return;
    wrapping = true;

    // Freeze all together
    for (const v of videos) v.pause();

    // Seek all together (to a safe non-zero start)
    for (const v of videos) seekSafe(v, LOOP_START);

    // Hold, then restart all together
    setTimeout(() => {
      Promise.resolve(master.play()).catch(() => {});
      for (const v of videos) {
        if (v === master) continue;
        v.play().catch(() => {});
      }
      wrapping = false;

      // Ensure the tick continues even if pause stopped callbacks
      requestAnimationFrame(tick);
    }, HOLD_MS);
  }

  function tick() {
    if (!started) return;

    const t = master.currentTime;

    // Wrap early (never let the video reach "ended")
    if (loopDur > 0 && !wrapping && t >= (loopDur - WRAP_EPS)) {
      wrapWithHold();
    }

    // Keep ticking (use rAF so it continues through pauses)
    requestAnimationFrame(tick);
  }

  async function start() {
    if (started) return;
    started = true;

    // Compute shared duration (shortest)
    const durs = videos.map(v => v.duration).filter(Number.isFinite);
    loopDur = Math.min(...durs);

    // Align start
    for (const v of videos) {
      v.loop = false;
      v.muted = true;
      v.playsInline = true;
      v.preload = "auto";
      seekSafe(v, LOOP_START);
    }

    try { await master.play(); } catch {}
    for (const v of videos) {
      if (v === master) continue;
      try { await v.play(); } catch {}
    }

    tick();
  }

  // Start when all can play
  let ready = 0;
  for (const v of videos) {
    if (v.readyState >= 2) {
      if (++ready === videos.length) start();
    } else {
      v.addEventListener("canplay", () => {
        if (++ready === videos.length) start();
      }, { once: true });
    }
  }
}

// Usage: group by data-scene like you already do
document.addEventListener("DOMContentLoaded", () => {
  const section = document.querySelector("#video-results");
  if (!section) return;

  const vids = [...section.querySelectorAll("video")];
  const groups = {};
  for (const v of vids) {
    const scene = v.dataset.scene || "default";
    (groups[scene] ||= []).push(v);
  }

  for (const videos of Object.values(groups)) {
    makeGroupLooper(videos);
  }
});







document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".dataset-dropdown").forEach((dropdown) => {
    const button = dropdown.querySelector(".dropdown-trigger button");
    if (!button) return;

    const setActive = (isActive) => {
      dropdown.classList.toggle("is-active", isActive);
      button.setAttribute("aria-expanded", String(isActive));
    };

    button.addEventListener("click", (e) => {
      e.preventDefault();
      setActive(!dropdown.classList.contains("is-active"));
    });

    document.addEventListener("click", (e) => {
      if (!dropdown.contains(e.target)) setActive(false);
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") setActive(false);
    });

    dropdown.querySelectorAll("a.dropdown-item").forEach((a) => {
      a.addEventListener("click", () => setActive(false));
    });
  });
});

function copyTextToClipboard(text) {
  // Prefer modern async clipboard when available
  if (navigator.clipboard && window.isSecureContext) {
    return navigator.clipboard.writeText(text);
  }

  // Fallback for older browsers / non-https contexts
  return new Promise((resolve, reject) => {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.top = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(ta);
      ok ? resolve() : reject(new Error("execCommand failed"));
    } catch (e) {
      reject(e);
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  const btn = document.querySelector(".bibtex-copy-btn");
  const codeEl = document.getElementById("bibtex-code");
  if (!btn || !codeEl) return;

  const defaultLabel = btn.innerHTML;

  btn.addEventListener("click", async () => {
    // innerText preserves line breaks; trim to avoid weird leading/trailing whitespace
    const bibtex = (codeEl.innerText || "").trim();

    try {
      await copyTextToClipboard(bibtex);

      btn.classList.add("is-success");
      btn.innerHTML = '<span class="icon is-small"><i class="fas fa-check"></i></span><span>Copied</span>';

      setTimeout(() => {
        btn.classList.remove("is-success");
        btn.innerHTML = defaultLabel;
      }, 1200);
    } catch (err) {
      btn.classList.add("is-danger");
      btn.innerHTML = '<span class="icon is-small"><i class="fas fa-times"></i></span><span>Failed</span>';

      setTimeout(() => {
        btn.classList.remove("is-danger");
        btn.innerHTML = defaultLabel;
      }, 1500);
    }
  });
});

document.addEventListener("DOMContentLoaded", () => {
  // Buttons/links that should open in a new tab IF they have a real URL
  document.querySelectorAll("a.js-external-btn").forEach((a) => {
    const href = (a.getAttribute("href") || "").trim();

    const isRealLink =
      href !== "" &&
      href !== "#" &&
      href.toLowerCase() !== "javascript:void(0)";

    if (isRealLink) {
      a.setAttribute("target", "_blank");
      a.setAttribute("rel", "noopener noreferrer");
      a.classList.remove("is-disabled");
      a.removeAttribute("aria-disabled");
      a.style.pointerEvents = "";
    } else {
      // No link yet: make it non-clickable and ensure no new tab behavior
      a.removeAttribute("target");
      a.removeAttribute("rel");
      a.setAttribute("aria-disabled", "true");
      a.classList.add("is-disabled");
      a.style.pointerEvents = "none";
    }
  });
});

