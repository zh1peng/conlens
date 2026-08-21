import { defineConfig } from 'vitepress'

const repository = 'https://github.com/zh1peng/conlens'
const siteUrl = 'https://zh1peng.github.io/conlens/'

export default defineConfig({
  base: '/conlens/',
  cleanUrls: true,
  lastUpdated: true,
  appearance: true,
  markdown: {
    math: true,
  },
  sitemap: {
    hostname: siteUrl,
  },
  head: [
    ['link', { rel: 'icon', type: 'image/png', href: '/conlens/conlens-logo.png' }],
    ['meta', { name: 'theme-color', content: '#0b3b66' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:image', content: `${siteUrl}conlens-logo.png` }],
  ],
  locales: {
    root: {
      label: '简体中文',
      lang: 'zh-CN',
      title: 'ConLens',
      titleTemplate: ':title · ConLens',
      description: '连接组全排序富集、leading-edge 网络与全流程 bootstrap 稳定性分析',
      themeConfig: {
        logo: '/conlens-mark.svg',
        siteTitle: 'ConLens',
        nav: [
          { text: '开始', link: '/guide/introduction' },
          { text: '教程', link: '/tutorials/design-and-contrasts' },
          { text: '结果解释', link: '/guide/interpretation' },
          { text: 'API', link: '/reference/api' },
        ],
        sidebar: [
          {
            text: '从这里开始',
            items: [
              { text: '软件包概览', link: '/' },
              { text: '认识 ConLens', link: '/guide/introduction' },
              { text: '安装', link: '/guide/installation' },
              { text: '五分钟快速开始', link: '/guide/quick-start' },
            ],
          },
          {
            text: '概念、推断与解释',
            items: [
              { text: '数据与 edge sets', link: '/guide/data-and-sets' },
              { text: '推断与零模型', link: '/guide/inference' },
              { text: '结果与 leading edge', link: '/guide/results' },
              { text: '如何解释结果', link: '/guide/interpretation' },
            ],
          },
          {
            text: '分析教程',
            items: [
              { text: '边统计量输入', link: '/tutorials/edge-statistics' },
              { text: 'Design matrix 与 contrasts', link: '/tutorials/design-and-contrasts' },
              { text: 'Bootstrap 稳定性', link: '/tutorials/stability' },
            ],
          },
          {
            text: '参考',
            items: [
              { text: 'Python API', link: '/reference/api' },
              { text: '命令行工具', link: '/reference/cli' },
            ],
          },
        ],
        outline: { level: [2, 3], label: '本页目录' },
        docFooter: { prev: '上一页', next: '下一页' },
        lastUpdated: { text: '最后更新于' },
        editLink: {
          pattern: `${repository}/edit/main/website/:path`,
          text: '在 GitHub 上编辑此页',
        },
        sidebarMenuLabel: '目录',
        returnToTopLabel: '返回顶部',
        langMenuLabel: '切换语言',
        darkModeSwitchLabel: '外观',
        search: { provider: 'local' },
        socialLinks: [{ icon: 'github', link: repository }],
        footer: {
          message: '以 MIT 许可证发布 · 文档以中文为主，英文版持续完善中',
          copyright: 'Copyright © ConLens contributors',
        },
      },
    },
    en: {
      label: 'English',
      lang: 'en-US',
      link: '/en/',
      title: 'ConLens',
      titleTemplate: ':title · ConLens',
      description: 'Ranked connectome enrichment and leading-edge networks',
      themeConfig: {
        logo: '/conlens-mark.svg',
        siteTitle: 'ConLens',
        nav: [
          { text: 'English home', link: '/en/' },
          { text: '中文文档', link: '/' },
          { text: 'GitHub', link: repository },
        ],
        sidebar: false,
        outline: false,
        search: { provider: 'local' },
        socialLinks: [{ icon: 'github', link: repository }],
        footer: {
          message: 'English documentation is under development.',
          copyright: 'Copyright © ConLens contributors',
        },
      },
    },
  },
})
