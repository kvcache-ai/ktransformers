import { createRouter, createWebHashHistory, RouteRecordRaw, createWebHistory } from 'vue-router'
import HomeView from '@/views/home.vue'

const routes: Array<RouteRecordRaw> = [
  {
    path: '/',
    name: 'home',
    component: HomeView,
    redirect: '/chat',
    children: [{
      path: '/chat',
      name: '',
      component: () => import(/* webpackChunkName: "about" */ '../components/chat/index.vue')
    },]
  },

]

const router = createRouter({
  history: createWebHashHistory(),
  routes
})

export default router
