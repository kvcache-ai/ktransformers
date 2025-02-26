<template>
  <div class="app-container" @contextmenu.prevent.stop="">
    <keep-alive>
      <router-view/>
    </keep-alive>
  </div>
</template>

<script setup lang="ts">
</script>

<style lang="stylus">
  @import "assets/iconfont/iconfont.css"
  #app
  .app-container
    width: 100%
    height: 100%
    position: relative
</style>