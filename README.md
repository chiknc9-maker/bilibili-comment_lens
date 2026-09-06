# 评论透镜（Comment Lens）— B站评论区思辨分析工具

**项目定位**：针对 B站争议视频的评论区，生成中立的三层分析——争论各方在争什么、每一方最有力的论证是什么、被当作"垃圾"的评论垃圾在论证的哪一环（以及其中有没有被误杀的观点内核）。

**M0 当前进度**：视频评论抓取（含楼中楼）→ JSON ✅

## 目录结构

```
comment-lens/
├── fetch_comments.py      # M0：BV号 → 评论 JSON
├── requirements.txt
├── .env.example           # 可选：填 SESSDATA 提升抓取稳定性
└── data/                  # 抓取结果输出目录
```

## M0 用法

```powershell
# 用项目虚拟环境（首次使用先激活或直接用完整路径）
.venv\Scripts\python.exe fetch_comments.py BV1RhtV6iEt3
.venv\Scripts\python.exe fetch_comments.py "https://www.bilibili.com/video/BV1RhtV6iEt3/"

# 可选参数
--main-total 100    # 抓多少条主评论（默认100，热评排序，每页20）
--no-subs           # 不抓楼中楼（更快）
--sub-limit 20      # 每条主评论的楼中楼条数上限
--depth3-limit 3    # 深楼每条二级抓多少条"回复的回复"（三级，0=关闭）
--depth3-floor 10   # 主楼楼中楼≥此值才下钻三级（控制请求量）
--delay 0.4         # 请求间隔（秒），遇到风控就加大
```

## 踩过的坑（重要，务必阅读）

1. **代理**：B站是国内站，且会对境外 IP 出口（Clash 等代理节点）直接风控（412）。
   脚本默认**直连**（忽略系统代理）。确实需要代理时用 `--proxy http://127.0.0.1:7892`。
2. **请求头一致性**：无 cookie 的匿名请求**不要带 Referer**（实测触发 412）；
   只有填了 cookie 脚本才会带 Referer + Cookie。脚本已自动处理，不要手动改。
3. **请求节奏**：默认每请求间隔 0.4s，抓 100 条主评论约需 1~2 分钟。
   连续高频请求会触发风控，风控后等几分钟再试。
4. **reply 接口的两层语义**：
   - `root=主评论rpid` → 该主楼的全部直接回复（每项含 rpid/parent/root）
   - `root=二级rpid` → 以该二级为根的**整棵对话树**（扁平展开，含二级自身
     与更深层），接口返回项的 rcount 字段**恒为 0**（B站已废弃该信号），
     判断是否有三级只能靠"下钻看返回"。
   - 三级评论 = 下钻返回里 `parent == 二级rpid 且 rpid != 二级rpid` 的项

## 遇风控怎么办（code=-412 / HTTP 412）

1. 等 5~10 分钟再试，或加大 `--delay 1.0`
2. 在 `.env` 中填入自己的 `SESSDATA`（复制 `.env.example` 改名 `.env`），
   登录态下风控阈值高得多，还能看更多评论

## JSON 数据格式

```jsonc
{
  "meta": {           // 视频信息与抓取配置
    "bvid", "aid", "title", "owner", "stat", "fetched_at", "config"
  },
  "threads": [        // 主评论（热评排序）
    {
      "rpid": 316116547328,   // 评论ID
      "uname": "...",         // 用户名（分析展示阶段将匿名化）
      "like": 5132,           // 点赞数
      "ctime": "ISO时间",
      "rcount": 57,           // 楼中楼总回复数（>实际抓到的数量=还有更深层争论）
      "text": "评论内容",
      "replies": [            // 楼中楼回复（按热度最多20条）
        { "rpid", "uname", "like", "ctime", "rcount", "text" }
      ]
    }
  ]
}
```

**伦理红线（产品原则）**：抓取仅用于本地分析。展示/分析阶段所有评论一律匿名化，
报告只评"论证"不评"人"。不做评分榜、不做公开排行。

## 路线图

- [x] M0：抓取脚本（BV → 楼中楼 JSON）
- [ ] M1：LangGraph 分析管线（立场地图 / 垃圾解剖 / 最佳论证 / 可疑事实）
- [ ] M2：FastAPI + 网页（粘贴链接 → 报告）
- [ ] M3：真实争议视频人工评测迭代

## 技术栈

- Python 3.10+（本机 3.14 可用）
- 直接调用 B站公开接口（`api.bilibili.com/x/v2/reply/main|reply`，hot 排序），
  未用第三方封装库（实测 bilibili-api-python 新版封装有坑，直连接口更可控）
- 后续 M1 起接入 LLM 分析管线
