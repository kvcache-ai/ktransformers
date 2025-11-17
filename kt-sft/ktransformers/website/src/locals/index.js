// index.js
import { createI18n } from 'vue-i18n'
import zh from './zh'
import en from './en'

const messages = {
  en,
  zh,
}
const language = (navigator.language || 'en').toLocaleLowerCase() // 这是获取浏览器的语言
const i18n = createI18n({
  legacy: false, // you must set `false`, to use Compostion API
  locale: localStorage.getItem('lang') || language.split('-')[0] || 'en', // 首先从缓存里拿，没有的话就用浏览器语言，
  fallbackLocale: 'en', // 设置备用语言
  messages, 
})

export default i18n