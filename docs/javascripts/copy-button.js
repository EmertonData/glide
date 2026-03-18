// Add copy button to code blocks
document.addEventListener('DOMContentLoaded', function() {
  // Find all code blocks
  const codeBlocks = document.querySelectorAll('pre');

  codeBlocks.forEach(block => {
    // Create copy button
    const button = document.createElement('button');
    button.className = 'copy-button';
    button.innerHTML = '<span class="copy-text">Copy</span><svg class="copy-icon" viewBox="0 0 24 24" width="18" height="18"><path fill="currentColor" d="M19 2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 16H5V4h14v14zm-3-7H8v2h8v-2z"/></svg>';
    button.title = 'Copy to clipboard';

    // Add copy functionality
    button.addEventListener('click', function() {
      const code = block.querySelector('code').textContent;
      navigator.clipboard.writeText(code).then(function() {
        // Show success feedback
        const originalHTML = button.innerHTML;
        button.innerHTML = '<span class="copy-text">Copied</span><svg class="copy-icon" viewBox="0 0 24 24" width="18" height="18"><path fill="currentColor" d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/></svg>';
        button.classList.add('copied');
        setTimeout(function() {
          button.innerHTML = originalHTML;
          button.classList.remove('copied');
        }, 2000);
      });
    });

    // Create wrapper and add button
    const wrapper = document.createElement('div');
    wrapper.className = 'code-block-wrapper';
    block.parentNode.insertBefore(wrapper, block);
    wrapper.appendChild(block);
    wrapper.appendChild(button);
  });
});
