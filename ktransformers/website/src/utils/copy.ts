import { ElMessage } from "element-plus";
const copy = (value: string) => {
  //Try using the navigator.clipboard.writeText method
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(value)
      .then(() => {
        //Using ElMessage to Display Success Messages in Windows Systems
        if (navigator.appVersion.includes("Win")) {
          ElMessage({
            message: "内容复制成功!",
            type: "success",
            plain: true,
          });
        } else {
          //Using custom DOM elements to display success messages in macOS system
          showCopySuccessMessage();
        }
      })
      .catch(() => {
        //Using ElMessage to Display Failure Messages in Windows Systems
        if (navigator.appVersion.includes("Win")) {
          ElMessage({
            message: "内容复制失败!",
            type: "error",
            plain: true,
          });
        } else {
          //Using custom DOM elements to display failure messages in macOS system
          showCopyErrorMessage();
        }
      });
  } else {
    const textarea = document.createElement("textarea");
    textarea.value = value;
    document.body.appendChild(textarea);
    textarea.select();
    try {
      const successful = document.execCommand('copy');
      if (successful) {
        if (navigator.appVersion.includes("Win")) {
          ElMessage({
            message: "内容复制成功!",
            type: "success",
            plain: true,
          });
        } else {
          showCopySuccessMessage();
        }
      } else {
        if (navigator.appVersion.includes("Win")) {
          ElMessage({
            message: "内容复制失败!",
            type: "error",
            plain: true,
          });
        } else {
          showCopyErrorMessage();
        }
      }
    } catch (err) {
      if (navigator.appVersion.includes("Win")) {
        ElMessage({
          message: "内容复制失败!",
          type: "error",
          plain: true,
        });
      } else {
        showCopyErrorMessage();
      }
    }
    document.body.removeChild(textarea);
  }
};

function showCopySuccessMessage() {
  const messageElement = document.createElement('div');
  messageElement.textContent = '内容复制成功!';
  messageElement.style.position = 'fixed';
  messageElement.style.bottom = '10px';
  messageElement.style.left = '50%';
  messageElement.style.transform = 'translateX(-50%)';
  messageElement.style.padding = '10px';
  messageElement.style.backgroundColor = '#4CAF50';
  messageElement.style.color = 'white';
  messageElement.style.borderRadius = '15px';
  messageElement.style.zIndex = '1000';
  document.body.appendChild(messageElement);
  setTimeout(() => {
    document.body.removeChild(messageElement);
  }, 3000);
}

function showCopyErrorMessage() {
  const messageElement = document.createElement('div');
  messageElement.textContent = '内容复制失败!';
  messageElement.style.position = 'fixed';
  messageElement.style.bottom = '10px';
  messageElement.style.left = '50%';
  messageElement.style.transform = 'translateX(-50%)';
  messageElement.style.padding = '10px';
  messageElement.style.backgroundColor = '#F44336';
  messageElement.style.color = 'white';
  messageElement.style.borderRadius = '5px';
  messageElement.style.zIndex = '1000';
  document.body.appendChild(messageElement);
  setTimeout(() => {
    document.body.removeChild(messageElement);
  }, 3000);
}

export default copy;