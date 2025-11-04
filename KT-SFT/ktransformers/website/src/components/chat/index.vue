<template>
  <div class="chat-panel">
    <!-- <div class="chat-model">{{ activeAssistant?.model }}</div> -->
    <div class="chat-panel-inner flex-column">
      <div class="chat-init flex-unit flex-column" v-if="isNotChating">
        <div class="assistant-info flex-column flex-unit">
          <div class="avatar">
            <img src="../../../public/images/avatar.png" />
          </div>
          <div class="name">
            {{ activeAssistant.name }}
          </div>
          <div class="desc">
            {{ activeAssistant.description }}
          </div>
        </div>
      </div>
      <div class="chat-msg flex-unit" v-else>
        <ul>
          <li
            class="chat-msg-item flex-row"
            v-for="(msg, index) in localMessages"
            :key="index"
          >
            <div class="avatar" v-if="msg.role == 'user'">
              <img src="../../../public/images/user-filling.png" />
            </div>
            <div class="avatar" v-else>
              <img src="../../../public/images/avatar.png" />
            </div>
            <div class="msg flex-unit">
              <div class="title flex-row">
                <div class="name">{{ msg.role }}</div>
                <div class="time flex-row">
                  {{ timeFormat(msg.created_at) }}
                </div>
              </div>
              <div
                class="content"
                v-html="markedText(msg.content)"
                ref="content_Ref"
              ></div>
              <div class="copy-btn flex-row" v-show="msgBttnBoxShow[index]">
                <i
                  class="iconfont icon-copy"
                  @click="copy(createText(msg.content))"
                ></i>
              </div>
            </div>
          </li>
        </ul>
      </div>
      <div class="scroll-box" v-show="showScrollButton" @click="scrollToBottom">
        <i class="iconfont icon-arrow-down"></i>
      </div>
      <div class="chat-send">
        <div
          class="chat-box flex-row"
          :style="{ height: textareaHeight + 'px' }"
          ref="chatBox_Ref"
        >
          <button @click="StopOutput" class="stop-btn" v-show="isRunning">
            stop
          </button>
          <textarea
            name="chat-input"
            class="chat-input flex-unit"
            :placeholder="inputPlaceholder"
            v-model="inputQuestion"
            @keydown="keyBoardCommitQuestion"
            :disabled="inputDisabled"
            :style="{ height: textareaHeight + 'px' }"
            @input="handleInput"
            ref="textarea_ref"
            maxlength="2000"
            cols="20"
          ></textarea>
          <i class="iconfont icon-sent" @click="clickCommitQuestion"></i>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import {
  defineComponent,
  nextTick,
  PropType,
  ref,
  watch,
  computed,
  onMounted,
} from "vue";
import { IThread, IMessageData, IAssistant } from "@/utils/types";
import { marked } from "marked";
import { createMessage } from "@/api/message";
import { createRun, cancelRun } from "@/api/run";
import { getAssistant } from "@/api/assistant";
import { createThread } from "@/api/thread";
import BScroll from "better-scroll";
import { useRouter, useRoute } from "vue-router";
import { useI18n } from "vue-i18n";
import { ElMessage } from "element-plus";
import { tr } from "element-plus/es/locale";
import copy from "@/utils/copy";
export default defineComponent({
  name: "ChatChat",
  props: {
    messages: {
      type: Array as PropType<IMessageData[]>,
      required: true,
    },
    chatInit: {
      type: Boolean,
      required: true,
    },
    activeAssistant: {
      type: Object as PropType<IAssistant>,
      required: true,
    },
    activeThread: {
      type: Object as PropType<IThread>,
      required: true,
    },
    inputDisabled: {
      type: Boolean,
      default: false,
    },
  },
  setup(props, context) {
    const { t } = useI18n();
    const router = useRouter();
    const route = useRoute();
    const localMessages = ref<IMessageData[]>([...props.messages]);
    const showScrollButton = ref(false);
    const messageScroll = ref<BScroll | null>(null);
    const inputQuestion = ref<string>("");
    const inputDisabled = ref(false);
    const msgBttnBoxShow = ref<boolean[]>([]);
    const answer = ref("");
    const activeThread = ref<IThread>({} as IThread);
    const activeAssistant = ref<IAssistant>({} as IAssistant);
    const isNotChating = ref(true);
    const isRunning = ref(false);
    const stopRunId = ref<string>("");
    const shouldContinueReceiving = ref(true);
    const textareaHeight = ref(48);
    const chatBox_Ref = ref();
    const textarea_ref = ref();
    const content_Ref = ref();
    // Boolean if go
    isNotChating.value = props.chatInit;
    activeThread.value = props.activeThread;
    activeAssistant.value = props.activeAssistant;
    watch(
      () => props.messages,
      (newMessages) => {
        localMessages.value = [...newMessages];
        msgBttnBoxShow.value = new Array(newMessages.length).fill(true);
      }
    );
    watch(
      () => props.inputDisabled,
      (newValue) => {
        inputDisabled.value = newValue;
      }
    );
    // Update scrollbars and scrolling events
    watch(
      () => localMessages.value,
      (newMessages) => {
        if (messageScroll.value) {
          scrollToTop();
          messageScroll.value.destroy();
          messageScroll.value = null;
        }
        if (!isNotChating.value) {
          nextTick(() => {
            messageScroll.value = new BScroll(".chat-msg", {
              click: true,
              mouseWheel: true,
              probeType: 3, //Only when set to 3 can the event of scrolling binding be triggered
            });
          });
        }
      },
      {
        immediate: true,
        deep: true,
      }
    );
    watch(
      () => messageScroll.value,
      (newValue) => {
        if (newValue) {
          messageScroll.value?.on("scroll", handleScroll);
          showScrollButton.value = false;
          scrollToBottom();
        }
      }
    );
    watch(
      () => props.chatInit,
      (newValue) => {
        isNotChating.value = newValue;
      }
    );
    watch(
      () => props.activeThread,
      (newValue) => {
        activeThread.value = newValue;
      }
    );
    watch(
      () => props.activeAssistant,
      (newValue) => {
        activeAssistant.value = newValue;
      }
    );

    const handleInput = (event:any) => {
      adjustHeight();
      const maxLength = 2000; 
      if (inputQuestion.value?.length > maxLength) {
        event.preventDefault(); 
        inputQuestion.value = inputQuestion.value.substring(0, maxLength); 
      }
    };
    const adjustHeight = () => {
      const currentScrollTop = textarea_ref.value.scrollTop;
      textarea_ref.value.style.height = textarea_ref.value.scrollHeight + "px";
      chatBox_Ref.value.style.height = textarea_ref.value.style.height;
      textarea_ref.value.scrollTop = currentScrollTop;
    };

    const inputPlaceholder = computed(() => {
      if (typeof activeAssistant.value.name != "undefined") {
        return replaceAssistant(t("chat.inputTip"), activeAssistant.value.name);
      } else {
        return t("chat.inputTip");
      }
    });
    // Block events
    const StopOutput = async () => {
      shouldContinueReceiving.value = false;
      try {
        const response = await cancelRun(
          activeThread.value.id,
          stopRunId.value
        );
        if (!response.ok) {
          console.error("Failed to cancel run");
        }
      } catch (error) {
        console.error("Failed to cancel run:", error);
      }
    };
    // dialogue
    const commitQuestion: () => void = async () => {
      const question = inputQuestion.value;
      // If it came in by clicking on assistants without clicking on thread, or through preview
      if (Object.keys(activeThread.value).length == 0) {
        try {
          let res = {} as IThread;
          // If you click thread and do not select assistant
          if (route.name == "preview") {
            let metadata = {
              hidden: "true",
            };
            res = await createThread(undefined, undefined, metadata);
          } else {
            res = await createThread();
          }
          activeThread.value = res;
        } catch (err) {
          console.error(err);
        }
      }
      //If you click thread and do not select assistant
      else if (Object.keys(activeAssistant.value).length == 0) {
        try {
          const messageOfAssistant = props.messages.find(
            (message) => message.role === "assistant"
          );
          if (messageOfAssistant && messageOfAssistant.assistant_id) {
            const res = await getAssistant(messageOfAssistant.assistant_id);
            activeAssistant.value = res;
          }
        } catch (err) {
          console.error(err);
        }
      }
      if (question) {
        inputQuestion.value = "";
        textareaHeight.value = 48;
        // inputDisabled.value = true;
        isNotChating.value = false;
        isRunning.value = true;
        await createMessage(activeThread.value.id, question)
          .then((res: any) => {})
          .catch((err: any) => {
            ElMessage({
              type: "warning",
              message: "Request error",
            });
            return;
          });
        // Current message queue insertion issue
        localMessages.value.push({
          role: "user",
          content: [
            { type: "text", text: { value: question }, annotatons: [] },
          ],
          created_at: Date.now() / 1000,
        });
        msgBttnBoxShow.value.push(true);
        // Insert answer into the current message queue
        localMessages.value.push({
          role: "assistant",
          content: [{ type: "text", text: { value: "" }, annotatons: [] }],
          created_at: Date.now() / 1000,
        });
        msgBttnBoxShow.value.push(false);
        try {
          const asyncGenerator = createRun(
            {
              assistant_id: activeAssistant.value.id,
              stream: true,
            },
            activeThread.value.id
          );
          for await (const word of asyncGenerator) {
            if (!shouldContinueReceiving.value) {
              break;
            }
            if (word.length == 36) {
              stopRunId.value = word;
              console.log(stopRunId.value);
            } else {
              answer.value += word;
              const index = localMessages.value.length - 1;
              localMessages.value[index].content[0].text.value += word;
              if (answer.value.length <= 3) {
                localMessages.value[index].created_at = Date.now() / 1000;
              }
            }
          }
        } catch (err) {
          console.error(err);
        }
        shouldContinueReceiving.value = true;
        answer.value = "";
        inputDisabled.value = false;
        msgBttnBoxShow.value[msgBttnBoxShow.value.length - 1] = true;
        scrollToBottom();
        isRunning.value = false;
        context.emit("updateAssistant", true);
        textarea_ref.value.focus();
      }
    };
    // Keyboard event stabilization
    const keyBoardCommitQuestion = (event: any) => {
      const question = inputQuestion.value?.trim();
      if (event.keyCode === 13) {
        event.preventDefault();

        const cursorPosition = event.target.selectionStart;
        if ((event.metaKey || event.ctrlKey) && question) {
          event.target.value =
            event.target.value.substring(0, cursorPosition) +
            "\n" +
            event.target.value.substring(cursorPosition);
          event.target.selectionStart = event.target.selectionEnd =
            cursorPosition + 1;
          adjustHeight();
          return;
        }
        if (!question) {
          ElMessage({
            message: "Please enter the content!",
            type: "warning",
            plain: true,
          });
          return;
        }
        if (!isRunning.value) {
          commitQuestion();
          inputQuestion.value = "";
        }
      }
    };
    const clickCommitQuestion = () => {
      if (!isRunning.value && inputQuestion.value?.trim() != "") {
        commitQuestion();
        return;
      }
      ElMessage({
        message: "Please enter the content!",
        type: "warning",
        plain: true,
      });
    };
    //Bottom scrolling
    const scrollToBottom = () => {
      //If messageScroll. value exists
      if (messageScroll.value) {
        //Call the scrollTo method of messageScroll. value and scroll to the bottom
        messageScroll.value.scrollTo(0, messageScroll.value?.maxScrollY, 800);
      }
    };
    // Top scrolling
    const scrollToTop = () => {
      if (messageScroll.value) {
        messageScroll.value.scrollTo(0, messageScroll.value?.minScrollY, 800);
      }
    };
    // Handling rolling events
    const handleScroll = (pos: any) => {
      if (messageScroll.value) {
        const distanceToBottom =
          messageScroll.value.y - messageScroll.value.maxScrollY;
        showScrollButton.value = distanceToBottom > 100;
      }
    };
    // Replace characters

    function replaceAssistant(input: string, newString: string) {
      return input.replace(/assistant/g, newString);
    }
    // Extract the markup text to convert the passed in object array into an HTML string parsed by market.js
    const markedText = (content: object[]) => {
      let context = "";
      for (const item of content) {
        if ((item as { type: string }).type === "text") {
          context += ((item as { text: object }).text as { value: string })
            .value;
        }
      }
      return marked.parse(context);
    };
    // Extract text content
    const createText = (content: object[]) => {
      let context = "";
      for (const item of content) {
        if ((item as { type: string }).type === "text") {
          context += ((item as { text: object }).text as { value: string })
            .value;
        }
      }
      return context;
    };
    // Time formatting
    const timeFormat = (timestamp: number | undefined) => {
      if (!timestamp) {
        return "";
      }
      const date = new Date(timestamp * 1000);
      // Obtain various time sections
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, "0"); // The month starts from 0 and needs to be increased by 1, with zeros added
      const day = String(date.getDate()).padStart(2, "0"); // Zero padding
      const hours = String(date.getHours()).padStart(2, "0"); // Zero padding
      const minutes = String(date.getMinutes()).padStart(2, "0"); // Zero padding
      const seconds = String(date.getSeconds()).padStart(2, "0"); // Zero padding
      // Format as "YYYY-MM-DD HH: mm: ss"
      const formattedDate = `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
      return formattedDate;
    };
    onMounted(() => {
      adjustHeight();
    });
    return {
      inputQuestion,
      inputDisabled,
      msgBttnBoxShow,
      localMessages,
      textareaHeight,
      answer,
      StopOutput,
      isNotChating,
      handleInput,
      chatBox_Ref,
      adjustHeight,
      content_Ref,
      markedText,
      timeFormat,
      createText,
      inputPlaceholder,
      keyBoardCommitQuestion,
      clickCommitQuestion,
      messageScroll,
      showScrollButton,
      commitQuestion,
      scrollToBottom,
      scrollToTop,
      isRunning,
      copy,
      replaceAssistant,
      textarea_ref,
    };
  },
});
</script>

<style scoped lang="stylus">
@import '@/assets/css/mixins.styl';

.chat-panel {
  justify-content: center;
  display: flex;
  position: relative;
  height: 100%;

  .chat-model {
    font-size: 16px;
    font-weight: bold;
    position: absolute;
    top: 20px;
    left: 30px;
  }

  .chat-panel-inner {
    width: 920px;
    padding-top: 80px;
  }

  .chat-init {
    padding: 0 20px;

    .assistant-info {
      text-align: center;
      align-items: center;
      justify-content: center;

      .avatar img {
        width: 70px;
        height: 70px;
      }

      .name {
        margin: 40px 0;
        font-size: 20px;
        font-weight: bold;
      }

      .desc {
        color: $gray_40;
      }
    }

    .assistant-tips {
      margin-bottom: 80px;

      .tips-item {
        width: 44%;
        height: 70px;
        line-height: 70px;
        float: left;
        border: 1px solid $border_gray_light_normal;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 10px;
        padding: 0 20px;
        color: $gray_40;

        &:nth-child(odd) {
          margin-left: 4%;
          margin-right: 4%;
        }

        &:nth-child(even) {
          margin-right: 4%;
        }

        .tips-ops {
          display: none;
          width: 24px;
          height: 24px;
          line-height: 24px;
          border-radius: 4px;
          text-align: center;
          border: 1px solid $border_gray_light_normal;

          i {
            font-size: 20px;
          }
        }

        &:hover {
          cursor: pointer;
          background-color: $bg_gray_light_hover;

          .tips-ops {
            display: block;
            background-color: #FFFFFF;
          }
        }
      }
    }
  }

  .chat-msg {
    overflow-y: hidden;

    ul {
      li.chat-msg-item {
        margin-bottom: 40px;
        align-items: flex-start !important;
        // border: 1px solid;
        border-radius: 15px;
        padding: 20px;
        margin-right: 20px;
        background-color: #313344;
        box-shadow: 12.5px 12.5px 10px rgba(0, 0, 0, 0.035), 10px 10px 8px rgba(0, 0, 0, 0.07);

        .avatar {
          margin-right: 15px;
          width: 36px;
          height: 36px;

          img {
            width: 100%;
            height: 100%;
            border-radius: 25px;
          }
        }

        .msg {
          .title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
            height: 36px;
            line-height: 24px;

            .time {
              justify-content: center;
              // margin-bottom: 12px;
              line-height: 20px;
              font-size: 14px;
              color: $gray_80;
            }

            .name {
              color: #edf2ea;
              font-size: 16px;
              font-weight: bold;
              margin-right: 15px;
            }

            .tips {
              font-size: 14px;
              color: $gray_50;
            }
          }

          .content {
            max-width: 829px;
            color: #edf2ea;
            font-size: 14px;
            line-height: 20px;
            word-wrap: break-word;
            margin-bottom: 12px;
          }

          .copy-btn {
            margin-top: 10px;
            justify-content: left;

            i {
              font-size: 20px;
              color: $gray_70;

              &:hover {
                cursor: pointer;
                color: $gray_50;

                .tips-ops {
                  display: block;
                  background-color: #FFFFFF;
                }
              }
            }
          }
        }
      }
    }
  }

  .chat-send {
    width: 900px;
    padding: 40px 0;
    position: relative;

    .chat-box {
      width: 100%;
      height: auto;
      min-height: 48px;
      max-height: 192px !important;
      border: none;
      border-radius: 15px;
      background: white;
      line-height: 48px;

      // overflow: hidden;
      .chat-input {
        height: auto;
        min-width: 900px;
        max-height: 192px !important;
        width: 100%;
        border: none;
        overflow-anchor: auto;
        overflow-x: hidden;
        overflow-y: auto;
        resize: none;
        background: white;
        display: inline-block;
      }

      .chat-input::-webkit-scrollbar {
        width: 10px;
      }

      .chat-input::-webkit-scrollbar-track {
        background-color: #f1f1f1;
      }

      .chat-input::-webkit-scrollbar-thumb {
        background-color: #888;
        border-radius: 5px;
      }

      .chat-input::-webkit-scrollbar-thumb:hover {
        background-color: #555;
      }

      .chat-input::-webkit-resizer {
        display: none;
      }

      .stop-btn {
        border: none;
        width: 60px;
        position: absolute;
        right: 50%;
        transform: translateX(50%);
        top: -40px;
        -webkit-border-radius: 50;
        -moz-border-radius: 50;
        border-radius: 50px;
        font-family: Arial;
        color: #ffffff;
        font-size: 16px;
        background: #cacdd1;
        padding: 10px 15px 10px 15px;
        text-decoration: none;
      }

      .stop-btn:hover {
        background: #8080e1;
        text-decoration: none;
        cursor: pointer;
      }
    }
  }
}

.scroll-box {
  position: absolute;
  bottom: 130px;
  right: 50%;
  transform: translateX(50%);
  margin: 0 auto;
  width: 32px;
  height: 32px;
  border-radius: 16px;
  border: 1px solid $gray_80;
  background-color: var(--el-bg-color-overlay);
  box-shadow: var(--el-box-shadow-lighter);
  text-align: center;
  line-height: 32px;
  color: #1989fa;

  i {
    font-size: 24px;
    color: $gray_60;
  }

  &:hover {
    cursor: pointer;
    background-color: $bg_gray_light_hover;

    i {
      color: $gray_50;
    }
  }
}
</style>