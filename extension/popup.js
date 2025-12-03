document.getElementById('summarizeBtn').addEventListener('click', async () => {
    const btn = document.getElementById('summarizeBtn');
    const loader = document.getElementById('loader');
    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error');
    const statusDiv = document.getElementById('status');

    // Reset UI
    btn.disabled = true;
    loader.style.display = 'block';
    resultDiv.style.display = 'none';
    errorDiv.textContent = '';
    statusDiv.textContent = 'Đang đọc nội dung bài báo...';

    try {
        // Get active tab
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

        // Execute script to get text
        const injectionResults = await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: extractContent,
        });

        const articleContent = injectionResults[0].result;

        if (!articleContent || articleContent.length < 100) {
            throw new Error("Không tìm thấy nội dung bài báo hoặc nội dung quá ngắn.");
        }

        statusDiv.textContent = 'Đang gửi đến Model để tóm tắt...';

        // Call API
        const response = await fetch('http://localhost:8000/summarize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: articleContent })
        });

        if (!response.ok) {
            throw new Error(`Lỗi server: ${response.statusText}`);
        }

        const data = await response.json();

        // Show result
        statusDiv.textContent = 'Hoàn tất!';
        resultDiv.textContent = data.summary;
        resultDiv.style.display = 'block';

    } catch (err) {
        errorDiv.textContent = err.message;
        statusDiv.textContent = 'Có lỗi xảy ra.';
    } finally {
        btn.disabled = false;
        loader.style.display = 'none';
    }
});

// Function to be executed in the page context
function extractContent() {
    // Try to find content in VnExpress specific classes
    // Common classes: .fck_detail, .Normal

    // Strategy 1: Get all paragraphs with class 'Normal' (Standard VnExpress)
    const paragraphs = document.querySelectorAll('p.Normal');
    if (paragraphs.length > 0) {
        return Array.from(paragraphs).map(p => p.innerText).join(' ');
    }

    // Strategy 2: Fallback to all paragraphs in the article body
    const articleBody = document.querySelector('article.fck_detail');
    if (articleBody) {
        return articleBody.innerText;
    }

    // Strategy 3: Generic fallback (get all paragraphs)
    return Array.from(document.querySelectorAll('p')).map(p => p.innerText).join(' ');
}
