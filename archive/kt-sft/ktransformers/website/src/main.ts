import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import store from './store'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import VueApexCharts from "vue3-apexcharts"
import i18n from '@/locals'

const app = createApp(App)

app.use(ElementPlus)

app.use(i18n)
app.use(VueApexCharts)
app.use(store)
app.use(router)
app.mount('#app')
