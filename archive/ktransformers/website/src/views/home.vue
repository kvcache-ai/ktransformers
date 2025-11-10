<template>
  <div class="home flex-row">
    <nav class="left-panel flex-column">
      <div class="logo-box">
        <div class="logo flex-row">
          <img class="img" src="../../public/images/three.png" />
          <span class="text">{{ projectName }}</span>
        </div>
        <div class="version">{{ projectVersion }}</div>
      </div>
      <div class="divider"></div>
      <div class="assistant-box">
        <div class="assistant-list">
          <ul>
            <li
              class="assistant-item flex-row"
              v-for="(item, index) in assistantList"
              :key="index"
              @click="setActiveAssistant(item)"
            >
              <img src="../../public/images/avatar.png" />
              <span class="name flex-unit">{{ item.name }}</span>
              <i class="iconfont icon-edit"></i>
            </li>
          </ul>
        </div>
      </div>
      <div class="divider"></div>
      <!-- History area -->
      <div class="history-box flex-unit">
        <div class="">
          <div class="date">{{ $t("home.today") }}</div>
          <ul>
            <li
              v-for="(item, index) in todayThreads"
              :key="index"
              class="chat-item"
              :class="{ active: activeThreadIndex === index }"
              @click="setActiveThreadIndex(index)"
            >
              <div class="chat-abbr">
                {{ firstMessages[index] }}
              </div>
              <div class="chat-ops flex-row">
                <img src="../../public/images/avatar.png" />
                <div class="name flex-unit">
                  {{ assistantOfThread[index].name || "" }}
                </div>
                <i class="iconfont icon-delete" @click="delThread(index)"></i>
              </div>
            </li>
          </ul>
          <div class="date" v-if="previousThreads.length > 0">
            {{ $t("home.previous") }}
          </div>
          <ul>
            <li
              v-for="(item, index) in previousThreads"
              :key="index"
              class="chat-item"
              :class="{
                active: activeThreadIndex === index + todayThreads.length,
              }"
              @click="setActiveThreadIndex(index + todayThreads.length)"
            >
              <div class="chat-abbr">
                {{ firstMessages[index + todayThreads.length] }}
              </div>
              <div class="chat-ops flex-row">
                <img src="../../public/images/avatar.png" />
                <div class="name flex-unit">
                  {{
                    assistantOfThread[index + todayThreads.length].name || ""
                  }}
                </div>
                <i
                  class="iconfont icon-delete"
                  @click="delThread(index + todayThreads.length)"
                ></i>
              </div>
            </li>
          </ul>
        </div>
      </div>
      <div class="icon-box example-2">
        <div class="iconhub icon-content" @click="navigateToIconHub">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            fill="currentColor"
            class="bi bi-github"
            viewBox="0 0 16 16"
            xml:space="preserve"
          >
            <path
              d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8"
              fill="currentColor"
            ></path>
          </svg>
          <div class="tooltip">GitHub</div>
        </div>
        <div class="iconlanguage" @click="changeLanguage">
          <svg
            v-if="!flag"
            t="1719306572024"
            class="icon"
            viewBox="0 0 1024 1024"
            version="1.1"
            xmlns="http://www.w3.org/2000/svg"
            p-id="16849"
            data-spm-anchor-id="a313x.search_index.0.i21.366e3a81tz0TYS"
            width="18"
            height="18"
          >
            <path
              d="M64.064 768V192H448.64v64H127.936v192h320v64h-320v192h320v64H64.064z m511.872 0V192h64l256 447.68V192h64v576h-64l-256-447.168V768h-64z"
              p-id="16850"
              data-spm-anchor-id="a313x.search_index.0.i22.366e3a81tz0TYS"
              class="selected"
              fill="#000000"
            ></path>
          </svg>
          <svg
            v-else
            t="1719306494614"
            class="icon"
            viewBox="0 0 1024 1024"
            version="1.1"
            xmlns="http://www.w3.org/2000/svg"
            p-id="12325"
            width="18"
            height="18"
          >
            <path
              d="M1023.488 831.552h-96l-265.472-451.904c-8.96-12.8-16-25.344-21.44-37.888H638.08c2.176 12.992 3.2 40.128 3.2 81.408v408.32L576 836.928V256h101.568l257.024 445.632c14.592 20.992 23.232 34.368 25.92 40.128h1.6c-2.688-16.512-4.032-44.8-4.032-84.736v-399.36L1024 256l-0.512 575.552zM435.008 804.224c-42.752 21.76-96.384 32.64-160.896 32.64-83.2 0-149.76-25.6-199.488-76.736C24.896 708.928 0 641.344 0 557.12c0-90.432 27.968-163.2 84.032-218.368C140.032 283.52 211.072 256 297.344 256c55.552 0 101.376 7.616 137.6 22.848v75.84a284.992 284.992 0 0 0-136.832-33.408c-64.768 0-117.504 20.864-158.208 62.592-40.768 41.728-61.184 98.048-61.184 168.96 0 67.2 19.008 120.576 57.024 160.128 38.016 39.552 87.744 59.328 149.248 59.328 57.536 0 107.52-12.544 150.016-37.76v69.696z"
              fill="#000000"
              p-id="12326"
              data-spm-anchor-id="a313x.search_index.0.i16.366e3a81tz0TYS"
              class="selected"
            ></path>
          </svg>
        </div>
      </div>
    </nav>
    <router-view v-slot="{ Component }" class="main-panel flex-unit">
      <component
        :is="Component"
        :chatInit="chatInit"
        :activeAssistant="activeAssistant"
        :activeThread="activeThread"
        :messages="allMessageInCurrentThread"
        :completedAssistant="assistantList"
        :inputDisabled="inputDisabled"
        @updateAssistant="handleUpdateAssistant"
      />
    </router-view>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, onMounted, computed, nextTick } from "vue";
import {
  IThread,
  IAssistant,
  IMessageData,
  IThreadAndMessageAndAssistant,
  IAssistantWithStatus,
} from "@/utils/types";
import { listThreads, deleteThread, getThread } from "@/api/thread";
import { ElMessage, ElMessageBox } from "element-plus";
import { listAssistants } from "@/api/assistant";
import { listMessages } from "@/api/message";
import { useRouter } from "vue-router";
import BScroll from "better-scroll";
import { useI18n } from "vue-i18n";

export default defineComponent({
  name: "HomeView",
  setup() {
    const assistantList = ref<IAssistant[]>([]);
    const threadsList = ref<IThread[]>([]);
    const firstMessages = ref<string[]>([]);
    const activeAssistant = ref({} as IAssistant);
    const assistantOfThread = ref<IAssistantWithStatus[]>([]);
    const threadAndMessages = ref<IThreadAndMessageAndAssistant[]>([]);
    const assistantScroll = ref<BScroll | null>(null);
    const historyScroll = ref<BScroll | null>(null);
    const router = useRouter();
    const { t, locale } = useI18n();
    const flag = ref(true);
    const changeLanguage = () => {
      if (flag.value) {
        locale.value = "zh";
        localStorage.setItem("lang", "zh");
        flag.value = false;
      } else {
        locale.value = "en";
        flag.value = true;
        localStorage.setItem("lang", "en");
      }
    };
    // Initialize data
    const initData = async () => {
      try {
        threadsList.value = [];
        firstMessages.value = [];
        assistantOfThread.value = [];

        const assistantsRes = await listAssistants();
        if (assistantsRes && assistantsRes.length > 0) {
          assistantList.value = assistantsRes;
          activeAssistant.value = assistantsRes[0];
        }

        const threadsRes = await listThreads(100);
        if (threadsRes) {
          threadAndMessages.value = threadsRes;
          for (let t of threadsRes) {
            if (t.thread && !t.thread.metadata?.hidden) {
              threadsList.value.push(t.thread);
              if (
                t.first_message &&
                t.first_message.content &&
                t.first_message.content.length > 0
              ) {
                firstMessages.value.push(t.first_message.content[0].text.value);
              } else {
                firstMessages.value.push("no message yet");
              }
              assistantOfThread.value.push(
                t.assistant || ({} as IAssistantWithStatus)
              );
            }
          }
        }

        assistantScroll.value = new BScroll(".assistant-list", {
          click: true,
          mouseWheel: true,
          scrollbar: {
            fade: true,
            interactive: true,
          },
        });

        historyScroll.value = new BScroll(".history-box", {
          click: true,
          mouseWheel: true,
          scrollbar: {
            fade: true,
            interactive: true,
          },
        });
      } catch (err) {
        console.error("Failed to initialize data:", err);
      }
    };
    const navigateToIconHub = () => {
      window.open("https://github.com/kvcache-ai/Lexllama");
    };
    const isEmptyObject = (obj: object): boolean => {
      //Determine if the object is empty
      return Object.keys(obj).length === 0;
    };
    //Jump route
    const navigateToExplore = () => {
      router.push("/explore");
    };
    const navigatorToChat = () => {
      router.push("/chat");
    };
    // Calculate date
    const todayThreads = computed(() => {
      const today = Math.floor(Date.now() / 1000);
      return threadsList.value.filter((thread) => {
        return today - thread.created_at <= 86400;
      });
    });
    const previousThreads = computed(() => {
      const today = Math.floor(Date.now() / 1000);
      return threadsList.value.filter((thread) => {
        return today - thread.created_at > 86400;
      });
    });

    onMounted(async () => {
      initData();
    });

    return {
      t,
      flag,
      assistantList,
      isEmptyObject,
      activeAssistant,
      navigateToExplore,
      navigatorToChat,
      threadsList,
      firstMessages,
      navigateToIconHub,
      assistantScroll,
      historyScroll,
      assistantOfThread,
      changeLanguage,
      initData,
      todayThreads,
      previousThreads,
    };
  },
  data() {
    return {
      projectName: "KTransformers",
      projectVersion: "v0.01",
      activeThreadIndex: -1,
      chatInit: true,
      activeThread: {} as IThread,
      allMessageInCurrentThread: [] as IMessageData[],
      inputDisabled: false,
      isSettingActiveThread: false,
      isDeletingThread: false,
      threadAndMessages: <IThreadAndMessageAndAssistant[]>[],
    };
  },
  methods: {
    setActiveAssistant(assistant: IAssistant) {
      this.chatInit = true;
      this.inputDisabled = false;
      this.activeThreadIndex = -1;
      this.activeAssistant = assistant;
      this.activeThread = {} as IThread;
      this.allMessageInCurrentThread = [];
      if (this.$route.path != "/chat") {
        this.navigatorToChat();
      }
    },
    async setActiveThreadIndex(index: number) {
      //If setting up an active thread, return directly
      if (this.isSettingActiveThread) {
        return;
      }
      this.isSettingActiveThread = true;
      this.activeThreadIndex = index;
      this.chatInit = false;
      this.inputDisabled = false;
      this.activeAssistant = {} as IAssistant;
      this.activeThread = this.threadsList[index];
      //If the assistant of the current thread is an empty object
      if (this.isEmptyObject(this.assistantOfThread[index])) {
        ElMessage({
          message: this.t("home.withoutAssistantTip"),
          type: "warning",
        });
        this.inputDisabled = true;
      }
      try {
        //Call asynchronous function to obtain the message list of the current thread
        const res = await listMessages(this.activeThread.id, 100, "asc");
        //Convert the obtained message list to the specified format and assign values to all messages of the current thread
        this.allMessageInCurrentThread = res.map((m) => ({
          role: m.role,
          content: m.content,
          assistant_id: m.assistant_id,
          created_at: m.created_at,
        }));
      } catch (err) {
        console.log(err);
      } finally {
        this.isSettingActiveThread = false;
      }
      if (this.$route.path != "/chat") {
        this.navigatorToChat();
      }
    },

    async delThread(index: number) {
      // If the thread is currently being deleted, return directly
      if (this.isDeletingThread) {
        return;
      }
      this.isDeletingThread = true;
      try {
        //Pop up a confirmation box and ask the user if they are sure to delete the thread
        await ElMessageBox.confirm(this.t("home.deleteThreadTip"), "Warning", {
          confirmButtonText: "OK",
          cancelButtonText: "Cancel",
          type: "warning",
        });

        const res = await deleteThread(this.threadsList[index].id);
        this.threadsList.splice(index, 1);
        this.firstMessages.splice(index, 1);
        this.assistantOfThread.splice(index, 1);
        // Jump to the first assistant or other suitable page
        this.setActiveAssistant(this.assistantList[0]);
        ElMessage({
          type: "success",
          message: "Delete completed",
        });
      } catch (err) {
        // Specific error handling, such as logging or displaying specific error messages to users
        console.error("Delete session failed:", err);
        ElMessage({
          type: "error",
          message: `Delete failed`, // Display specific error messages
        });
      } finally {
        this.isDeletingThread = false; //Ensure that the delete thread flag is reset no matter what
      }
    },
    // Handles the update of the assistant asynchronously.
    async handleUpdateAssistant(value: any) {
      await this.initData();
      if (this.activeThreadIndex != -1) {
        this.setActiveThreadIndex(this.activeThreadIndex);
      } else if (this.activeAssistant.id) {
        this.setActiveThreadIndex(0);
      } else {
        this.setActiveAssistant(this.assistantList[0]);
      }
    },
  },
});
</script>


<style lang="stylus" rel="stylesheet/stylus" scoped>
@import '../assets/css/mixins.styl';

.home {
  width: 100%;
  height: 100%;
  position: relative;
}

.left-panel {
  width: 320px;
  height: 100%;
  background-color: #363433;
  padding: 30px 30px;
  .logo-box {
    .logo {
      .img {
        width: 36px;
        height: 36px;
      }

      .text {
        font-size: 28px;
        font-weight: bold;
        margin-left: 10px;
        color: #edf2ea;
      }
    }

    .version {
      text-align: right;
      font-size: 14px;
      color: #bdbdbd;
    }
  }

  .divider {
    border-bottom: 1px solid #D7D7D7;
    width: 30%;
    margin: 30px auto;
  }

  .lang-box {
    position: relative;
    width: 100%;
    height: 30px;
    margin: auto;
    margin-bottom: 10px;

    .el-dropdown {
      font-size: 14px;
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
    }
  }

  .assistant-box {
    .assistant-list {
      min-height: 50px;
      max-height: 300px;
      overflow: hidden;
      position: relative;

      ul > li.assistant-item {
        padding: 8px 15px;
        color: #edf2ea;

        img {
          width: 32px;
          height: 32px;
        }

        .name {
          margin-left: 12px;
          font-size: 14px;
          color: #edf2ea;
        }

        i.iconfont {
          display: none;
          margin-left: 10px;
        }

        &:hover {
          background-color: $bg_gray_light_hover;
          cursor: pointer;
          border-radius: 4px;

          .name {
            color: #313433;
          }

          i.iconfont {
            display: block;
          }
        }
      }
    }

    .explore {
      position: relative;
      justify-content: center;
      display: flex;
      margin-top: 10px;

      .explore-btn {
        margin: 0 auto;
        padding: 0 20px;
        justify-content: center;
        height: 32px;
        line-height: 32px;
        background-color: #FFFFFF;
        border: 1px solid RGBA(0, 0, 0, 0.15);
        border-radius: 16px;

        i {
          color: #8080FF;
        }

        .text {
          color: #7F7F7F;
          margin-left: 4px;
        }

        &:hover {
          background-color: #FAFAFA;
          cursor: pointer;
        }
      }
    }
  }

  .history-box {
    position: relative;

    .date {
      font-size: 14px;
      color: #7F7F7F;
      margin: 8px 0;

      &:first-child {
        margin-top: 0;
      }
    }

    li.chat-item {
      padding: 12px 15px;
      cursor: pointer;
      background-color: #edf2ea;
      border-radius: 4px;
      margin-bottom: 10px;
      font-size: 16px;

      .chat-abbr {
        font-size: 14px;
        color: #313433;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .chat-ops {
        display: flex;
        margin-top: 5px;

        img {
          width: 16px;
          height: 16px;
        }

        .name {
          font-size: 12px;
          color: #898989;
          margin-left: 8px;
        }

        i.iconfont {
          color: $gray_60;
        }
      }

      &:hover, &.active {
        transition: 0.3s all;
        cursor: pointer;
        background-color: #a2a79f;
        .chat-abbr {
          color: black;
        }

        .name, i.iconfont {
          color: black;
        }
      }
    }
  }

  .icon-box {
    width: 100%;
    display: flex;
    flex-direction: row;
    justify-content: flex-end;
    align-items: center;

    .iconhub {
      width: 32px;
      height: 24px;
      background: white;
      font-size: 30px;
      border: none;
      ovferflow: hidden;
      border-radius: 15%;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      color: #898989;
      transition: all 0.5s;
      cursor: pointer;
    }

    .iconhub:hover {
      background: #e5e5e5;
      text-decoration: none;
    }

    .iconlanguage {
      margin-left: 15px;
      width: 32px;
      height: 24px;
      background: white;
      font-size: 30px;
      border: none;
      ovferflow: hidden;
      border-radius: 15%;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      color: #898989;
      transition: all 0.5s;
      cursor: pointer;
    }

    .iconlanguage:hover {
      background: #e5e5e5;
      text-decoration: none;
    }
  }
}

ul {
  list-style: none;
}

.example-2 {
  display: flex;
  justify-content: center;
  align-items: center;
}

.example-2 .icon-content {
  margin: 0 10px;
  position: relative;
}

.example-2 .icon-content .tooltip {
  position: absolute;
  top: -30px;
  left: 50%;
  transform: translateX(-50%);
  color: #fff;
  padding: 6px 10px;
  border-radius: 5px;
  opacity: 0;
  visibility: hidden;
  font-size: 14px;
  transition: all 0.3s ease;
}

.example-2 .icon-content:hover .tooltip {
  opacity: 1;
  visibility: visible;
  top: -50px;
}

.main-panel {
  height: 100%;
  background-color: #f1f0ed;
}
</style>
