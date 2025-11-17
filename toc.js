// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item affix "><a href="index.html">Introduction</a></li><li class="chapter-item affix "><li class="part-title">Install &amp; Usage</li><li class="chapter-item "><a href="en/kt-kernel/kt-kernel_intro.html"><strong aria-hidden="true">1.</strong> For kt-kernel</a></li><li class="chapter-item "><a href="en/SFT/KTransformers-Fine-Tuning_User-Guide.html"><strong aria-hidden="true">2.</strong> For kt-sft</a></li><li class="chapter-item affix "><li class="part-title">Tutorial</li><li class="chapter-item "><a href="en/SFT/index.html"><strong aria-hidden="true">3.</strong> kt-sft part</a><a class="toggle"><div>❱</div></a></li><li><ol class="section"><li class="chapter-item "><a href="en/SFT/KTransformers-Fine-Tuning_Developer-Technical-Notes.html"><strong aria-hidden="true">3.1.</strong> kt-sft developer tech notes</a></li><li class="chapter-item "><a href="en/SFT/injection_tutorial.html"><strong aria-hidden="true">3.2.</strong> Injection Tutorial</a></li></ol></li><li class="chapter-item "><a href="en/kt-kernel/index.html"><strong aria-hidden="true">4.</strong> kt-kernel part</a></li><li class="chapter-item affix "><li class="part-title">FAQ</li><li class="chapter-item "><a href="en/FAQ.html"><strong aria-hidden="true">5.</strong> FAQ</a></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split("#")[0].split("?")[0];
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
