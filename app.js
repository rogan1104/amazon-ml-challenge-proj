/**
 * Interactive Application Controller for Amazon ML Challenge 2026 Documentation
 */
document.addEventListener('DOMContentLoaded', () => {
  // Navigation elements
  const navItems = document.querySelectorAll('.nav-item');
  const docSections = document.querySelectorAll('.doc-section');
  const currentSectionCrumb = document.getElementById('currentSectionCrumb');
  const contentBody = document.getElementById('contentBody');
  const sidebar = document.getElementById('sidebar');
  const mobileOpenBtn = document.getElementById('mobileOpenBtn');
  const mobileCloseBtn = document.getElementById('mobileCloseBtn');
  const navSearchInput = document.getElementById('navSearchInput');

  // Theme Elements
  const themeToggleBtn = document.getElementById('themeToggleBtn');
  const themeText = themeToggleBtn.querySelector('.theme-text');

  // Next / Prev Buttons
  const nextSectionBtns = document.querySelectorAll('.next-section-btn');
  const prevSectionBtns = document.querySelectorAll('.prev-section-btn');
  const returnTopBtn = document.getElementById('returnTopBtn');
  const printDocsBtn = document.getElementById('printDocsBtn');

  // Copy Code Buttons
  const copyButtons = document.querySelectorAll('.copy-code-btn');

  // F0.5 Simulator Elements
  const precisionSlider = document.getElementById('precisionSlider');
  const recallSlider = document.getElementById('recallSlider');
  const precisionVal = document.getElementById('precisionVal');
  const recallVal = document.getElementById('recallVal');
  const calcF05 = document.getElementById('calcF05');
  const calcF10 = document.getElementById('calcF10');
  const calcInsight = document.getElementById('calcInsight');

  /**
   * Switch Active Section
   */
  function activateSection(targetId) {
    const targetSection = document.getElementById(targetId);
    if (!targetSection) return;

    // Update active class on sections
    docSections.forEach(sec => sec.classList.remove('active'));
    targetSection.classList.add('active');

    // Update active class on sidebar items
    navItems.forEach(item => {
      if (item.getAttribute('data-section') === targetId) {
        item.classList.add('active');
        const title = targetSection.getAttribute('data-title') || item.querySelector('.nav-title').textContent;
        currentSectionCrumb.textContent = title;
      } else {
        item.classList.remove('active');
      }
    });

    // Scroll to top of content
    contentBody.scrollTop = 0;

    // Close mobile menu if open
    if (sidebar.classList.contains('open')) {
      sidebar.classList.remove('open');
    }

    // Update URL Hash
    history.replaceState(null, null, `#${targetId}`);
  }

  // Bind Sidebar Item Clicks
  navItems.forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const targetId = item.getAttribute('data-section');
      activateSection(targetId);
    });
  });

  // Next / Previous buttons inside sections
  nextSectionBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-target');
      activateSection(targetId);
    });
  });

  prevSectionBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-target');
      activateSection(targetId);
    });
  });

  if (returnTopBtn) {
    returnTopBtn.addEventListener('click', () => {
      activateSection('section-1');
    });
  }

  // Handle Initial Hash or Default to Section 1
  const initialHash = window.location.hash.replace('#', '');
  if (initialHash && document.getElementById(initialHash)) {
    activateSection(initialHash);
  } else {
    activateSection('section-1');
  }

  // Mobile Drawer Toggle
  if (mobileOpenBtn) {
    mobileOpenBtn.addEventListener('click', () => {
      sidebar.classList.add('open');
    });
  }

  if (mobileCloseBtn) {
    mobileCloseBtn.addEventListener('click', () => {
      sidebar.classList.remove('open');
    });
  }

  // Print / Export
  if (printDocsBtn) {
    printDocsBtn.addEventListener('click', () => {
      window.print();
    });
  }

  // Search Navigation Filter
  if (navSearchInput) {
    navSearchInput.addEventListener('input', (e) => {
      const query = e.target.value.toLowerCase().trim();
      navItems.forEach(item => {
        const title = item.querySelector('.nav-title').textContent.toLowerCase();
        const desc = item.querySelector('.nav-desc').textContent.toLowerCase();
        if (!query || title.includes(query) || desc.includes(query)) {
          item.style.display = 'flex';
        } else {
          item.style.display = 'none';
        }
      });
    });

    // Keyboard shortcut to focus search with '/'
    document.addEventListener('keydown', (e) => {
      if (e.key === '/' && document.activeElement !== navSearchInput) {
        e.preventDefault();
        navSearchInput.focus();
        navSearchInput.select();
      }
    });
  }

  // Theme Toggle (Dark / Light)
  const savedTheme = localStorage.getItem('er-docs-theme') || 'dark';
  if (savedTheme === 'light') {
    document.body.classList.remove('dark-theme');
    document.body.classList.add('light-theme');
    themeText.textContent = 'Light Mode';
  }

  themeToggleBtn.addEventListener('click', () => {
    if (document.body.classList.contains('dark-theme')) {
      document.body.classList.remove('dark-theme');
      document.body.classList.add('light-theme');
      themeText.textContent = 'Light Mode';
      localStorage.setItem('er-docs-theme', 'light');
    } else {
      document.body.classList.remove('light-theme');
      document.body.classList.add('dark-theme');
      themeText.textContent = 'Dark Mode';
      localStorage.setItem('er-docs-theme', 'dark');
    }
  });

  // Code Block Copy Functionality
  copyButtons.forEach(btn => {
    btn.addEventListener('click', async () => {
      const codeId = btn.getAttribute('data-code-id');
      const codeElement = document.getElementById(codeId);
      if (!codeElement) return;

      const codeText = codeElement.innerText || codeElement.textContent;
      try {
        await navigator.clipboard.writeText(codeText);
        const originalText = btn.querySelector('span').textContent;
        btn.querySelector('span').textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(() => {
          btn.querySelector('span').textContent = originalText;
          btn.classList.remove('copied');
        }, 2000);
      } catch (err) {
        console.error('Clipboard copy failed:', err);
      }
    });
  });

  // Interactive F0.5 Simulator Logic
  function updateF05Simulator() {
    if (!precisionSlider || !recallSlider) return;

    const P = parseFloat(precisionSlider.value);
    const R = parseFloat(recallSlider.value);

    precisionVal.textContent = P.toFixed(2);
    recallVal.textContent = R.toFixed(2);

    // Official Formula: F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    let f05 = 0;
    const denom05 = (0.25 * P + R);
    if (denom05 > 0) {
      f05 = (1.25 * P * R) / denom05;
    }

    // Standard F1.0 Formula: (2 * P * R) / (P + R)
    let f10 = 0;
    const denom10 = (P + R);
    if (denom10 > 0) {
      f10 = (2 * P * R) / denom10;
    }

    calcF05.textContent = f05.toFixed(3);
    calcF10.textContent = f10.toFixed(3);

    const diff = f05 - f10;
    if (diff > 0.005) {
      calcInsight.innerHTML = `High precision pays off: <strong>F<sub>0.5</sub> (${f05.toFixed(3)})</strong> exceeds F<sub>1.0</sub> by <strong>+${diff.toFixed(3)}</strong>.`;
      calcInsight.style.borderColor = 'rgba(16, 185, 129, 0.4)';
    } else if (diff < -0.005) {
      calcInsight.innerHTML = `Low precision is severely penalized: <strong>F<sub>0.5</sub> (${f05.toFixed(3)})</strong> trails F<sub>1.0</sub> by <strong>${diff.toFixed(3)}</strong>. Increase precision!`;
      calcInsight.style.borderColor = 'rgba(244, 63, 94, 0.4)';
    } else {
      calcInsight.innerHTML = `Balanced: F<sub>0.5</sub> and F<sub>1.0</sub> are identical at ${f05.toFixed(3)}.`;
      calcInsight.style.borderColor = 'rgba(56, 189, 248, 0.4)';
    }
  }

  if (precisionSlider && recallSlider) {
    precisionSlider.addEventListener('input', updateF05Simulator);
    recallSlider.addEventListener('input', updateF05Simulator);
    updateF05Simulator();
  }
});
