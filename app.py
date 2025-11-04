# app_streamlit.py — 极简前端：先配对预览 → 用户选C/D → 弹出结果 → 刷新轮次
import streamlit as st
import random
import pandas as pd
import altair as alt
from pd_core import (
    Payoffs, Action,
    build_agents_without_bay_and_with_user, Simulator
)

# ---------- 页面与样式 ----------
st.set_page_config(page_title="Iterated Prisoner's Dilemma – USER", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
/* 隐藏侧边栏 */
[data-testid="stSidebar"]{display:none;}
section[data-testid="stSidebar"] + section{margin-left:0;}
/* 百分比条样式（蓝=C，红=D） */
.bar-wrap{width:100%;height:22px;border-radius:999px;overflow:hidden;background:#f0f2f6;border:1px solid #e3e6ef;}
.bar-C{height:100%;background:#3b82f6;display:inline-block;}
.bar-D{height:100%;background:#ef4444;display:inline-block;}
.bar-labels{display:flex;justify-content:space-between;font-size:12px;margin-top:6px;color:#4b5563;}
.small-muted{color:#6b7280;font-size:12px;}
/* 主操作区稍微紧凑些 */
.block-container{padding-top:1.2rem;}
</style>
""", unsafe_allow_html=True)


AGENT_NAME_CN = {
    "USER": "诸葛亮",
    "TFT": "关羽",
    "gTFT0.15": "张飞",
    "SG3": "黄忠",
    "ALT": "赵云",
    "R50": "刘备",
    "sWSLS20": "马超",
    "Joss10": "周瑜",
    "M1": "司马懿",
    "Gradual": "吕蒙",
    "AC": "鲁肃",
    "AD": "曹操",
    "GRIM": "张辽",
    "PROB": "吕布",
    "sTFT": "许褚",
    "WSLS": "陆逊",
    "TF2T": "庞统",
    "CTFT": "孙权",
    "Tester": "貂蝉",
    "Majority": "张郃",
}

def cn(name: str) -> str:
    """只改变显示，不改内部逻辑/名字"""
    return AGENT_NAME_CN.get(name, name)

# ---------- 固定默认参数（不对外展示） ----------
DEFAULTS = dict(
    seed=random.randint(0, 10000), delta=0.2,
    T=5, R=4, P=0, S=-1,
    gTFT_generosity=0.15, SG_punish=3, ALT_startC=1,
    R50_p=0.5, sWSLS_eps=0.2, Joss_p=0.1,
    M1_pCC=0.95, M1_pCD=0.2, M1_pDC=0.9, M1_pDD=0.1
)

# ---------- 初始化 ----------
def init_sim():
    agents, user = build_agents_without_bay_and_with_user(
        gTFT_generosity=DEFAULTS["gTFT_generosity"],
        SG_punish=DEFAULTS["SG_punish"],
        ALT_startC=DEFAULTS["ALT_startC"],
        R50_p=DEFAULTS["R50_p"],
        sWSLS_eps=DEFAULTS["sWSLS_eps"],
        Joss_p=DEFAULTS["Joss_p"],
        M1_pCC=DEFAULTS["M1_pCC"], M1_pCD=DEFAULTS["M1_pCD"],
        M1_pDC=DEFAULTS["M1_pDC"], M1_pDD=DEFAULTS["M1_pDD"],
    )
    sim = Simulator(
        agents,
        Payoffs(DEFAULTS["T"], DEFAULTS["R"], DEFAULTS["P"], DEFAULTS["S"]),
        seed=DEFAULTS["seed"], delta=DEFAULTS["delta"]
    )
    sim.reset()
    st.session_state.sim = sim
    st.session_state.user = user
    st.session_state.preview_pairs = None
    st.session_state.last_flash = None

if "sim" not in st.session_state:
    init_sim()

sim: Simulator = st.session_state.sim
user = st.session_state.user

# ---------- 工具函数 ----------
def ensure_preview_pairs(sim: Simulator):
    """
    优先使用 sim.preview_pairs() 生成“本轮预览配对”（无副作用）并缓存。
    若无该方法，则退回使用 last_pairs（第一轮可能没有）。
    缓存内容为 [("A","B"), ...] 的名字对。
    """
    if st.session_state.preview_pairs is not None:
        return st.session_state.preview_pairs

    pairs = None
    # 有预览方法：最稳
    if hasattr(sim, "preview_pairs"):
        try:
            p = sim.preview_pairs()
            pairs = [(a.name, b.name) for (a, b) in p]
        except Exception:
            pairs = None

    # 回退：用 last_pairs（注意第一轮可能为空）
    if pairs is None:
        if sim.last_pairs:
            pairs = [(a.name, b.name) for (a, b) in sim.last_pairs]
        else:
            pairs = []  # 无法预览

    st.session_state.preview_pairs = pairs
    return pairs

def current_opponent_for_user(preview_pairs):
    for a, b in preview_pairs:
        if a == "USER": return b
        if b == "USER": return a
    return None

def opponent_cd_percent_global(sim: Simulator, opp_name: str):
    """
    读取 sim.action_counts（全局计数）计算对手 C/D 百分比。
    若无数据返回 (None, None)。
    """
    if not opp_name:
        return None, None
    counts = getattr(sim, "action_counts", None)
    if not counts or opp_name not in counts:
        return None, None
    c = counts[opp_name].get("C", 0)
    d = counts[opp_name].get("D", 0)
    tot = c + d
    if tot <= 0:
        return None, None
    c_pct = max(0.0, min(100.0, 100.0 * c / tot))
    d_pct = 100.0 - c_pct
    return c_pct, d_pct

def render_cd_bar(c_pct, d_pct, opp_name):
    """显示对手合作/背叛比例条（绿色=合作，红色=背叛，加起来100%）"""
    if c_pct is None:
        st.markdown(
            f"<div class='small-muted'><b>{cn(opp_name)}</b> 还没有历史决策数据。</div>",
            unsafe_allow_html=True
        )
        st.markdown("""
        <div class='bar-wrap'>
            <span class='bar-C' style='width:0%;background:#16a34a;'></span>
            <span class='bar-D' style='width:100%;background:#dc2626;'></span>
        </div>
        <div class='bar-labels'><span>合作: --</span><span>背叛: --</span></div>
        """, unsafe_allow_html=True)
        return

    # 主体：合作和背叛两段拼成 100%
    html = f"""
    <div style="
        width:100%;
        height:22px;
        border-radius:999px;
        overflow:hidden;
        display:flex;
        border:1px solid #e3e6ef;">
        <div style="width:{c_pct:.2f}%;background:#16a34a;"></div>
        <div style="width:{d_pct:.2f}%;background:#dc2626;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:16px;margin-top:6px;color:#374151;">
        <span><b style='color:#16a34a;'>合作 🤝</b>: {c_pct:.1f}%</span>
        <span><b style='color:#dc2626;'>背叛 ⚔️</b>: {d_pct:.1f}%</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def extract_user_outcome(step_info):
    """
    从 step(info) 中抽取 USER 对局信息：
    返回: opp_name, my_action, opp_action, my_payoff
    期望 info["pairs"] 为 [(a1, m1, a2, m2, p1, p2), ...]
    其中 m1/m2 为 Action 或 'C'/'D'；若为枚举则显示时转字符串。
    """
    if not step_info or "pairs" not in step_info:
        return None, None, None, None
    for (a1, m1, a2, m2, p1, p2) in step_info["pairs"]:
        if a1 == "USER":
            return a2, (m1.value if hasattr(m1, "value") else str(m1)), (m2.value if hasattr(m2, "value") else str(m2)), p1
        if a2 == "USER":
            return a1, (m2.value if hasattr(m2, "value") else str(m2)), (m1.value if hasattr(m1, "value") else str(m1)), p2
    return None, None, None, None


def get_agent_by_name(sim: Simulator, name: str):
    """从模拟器里根据名字拿到真正的 Agent 对象"""
    for a in sim.agents:
        if a.name == name:
            return a
    return None

def render_last_action(user_agent, opp_agent):
    """
    显示这个对手上一次对 USER 的动作
    last_with 返回类似: (my_last, opp_last, payoff)
    """
    if user_agent is None or opp_agent is None:
        return

    my_last, opp_last, _ = user_agent.last_with(opp_agent)

    if opp_last is None:
        st.markdown(
            "<div style='font-size:18px;color:#6b7280;'>这个对手还没有和你打过一轮。</div>",
            unsafe_allow_html=True
        )
    else:
        # 根据你前面 Action 的写法，这里兼容枚举/字符串
        if hasattr(opp_last, "value"):
            opp_v = opp_last.value
        else:
            opp_v = str(opp_last)

        if opp_v.upper() == "C":
            txt = "合作 🤝"
            color = "#16a34a"
        else:
            txt = "背叛 ⚔️"
            color = "#dc2626"

        st.markdown(
            f"<div style='font-size:18px;'>上一次 <b style='color:#0f172a;'>{cn(opp_agent.name)}</b> 对你是："
            f"<b style='color:{color};'>{txt}</b></div>",
            unsafe_allow_html=True
        )


col_left_pad, col_main, col_right_pad = st.columns([2, 4, 2])
with col_left_pad:
    st.markdown("<div class='pad-col' style='border-right:1px solid #e5e7eb;'></div>", unsafe_allow_html=True)

with col_right_pad:
    st.markdown("<div class='pad-col' style='border-left:1px solid #e5e7eb;'></div>", unsafe_allow_html=True)
with col_main:
# ---------- 页面 ----------
    st.markdown(
        "<h1 style='text-align:center; font-size:38px; font-weight:900; color:#1e293b;'>三国争霸小游戏 ⚔️</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; font-size:20px; color:#64748b;'>"
        "游戏说明：你扮演的是 <span style='color:#dc2626; font-weight:600;'>诸葛亮</span>，"
        "请选择你的策略，目标在第100天时的收益排在第一名🏆"
        "</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # 上次结果的简短提示（若需要）
    if st.session_state.get("last_flash"):
        try:
            st.toast(st.session_state.last_flash)
        except Exception:
            st.info(st.session_state.last_flash)
    st.session_state.last_flash = None

    left, right = st.columns([1.3, 1.0])

    with left:
        st.markdown(
            f"<h3 style='font-size26px; font-weight:700; color:#1e293b;'>"
            f"天数: <span style='color:#2563eb;'>{sim.round + 1}</span>"
            f"</h3>",
            unsafe_allow_html=True
        )

        # 先生成“本轮预览配对”（第一轮也会尝试得到）
        preview_pairs = ensure_preview_pairs(sim)
        opp_name = current_opponent_for_user(preview_pairs)

        if opp_name:
            st.markdown(
                f"<div style='font-size:26px; font-weight:700; color:#1e293b;'>匹配到的对手：<span style='color:#2563eb;'>{cn(opp_name)}</span></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='font-size:22px; color:#6b7280;'>尚未匹配到对手</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")  # ✅ 横线分割
        st.markdown("### 请选择你的策略")
        # 中文显示映射
        action_labels = {
            Action.C.value: "合作 🤝",
            Action.D.value: "背叛 ⚔️"
        }
        # 说明：合作/背叛的得分机制
        st.markdown("""
        <div style='
            background-color:#f8fafc;
            border-left:5px solid #2563eb;
            padding:10px 16px;
            margin-top:10px;
            font-size:15px;
            color:#334155;
        '>
        <b>规则说明：</b><br>
        当你与对手同时合作 🤝 → 双方各得 <b>4 分</b>；<br>
        若你背叛 ⚔️ 而对方合作 → 你得 <b>5 分</b>，对方减 <b>1 分</b>；<br>
        若双方都背叛 ⚔️ → 各得 <b>0 分</b>；<br>
        若你合作 🤝 而对方背叛 ⚔️ → 你减 <b>1 分</b>，对方得 <b>5 分</b>。
        </div>
        """, unsafe_allow_html=True)
        choice_label = st.radio(
            "Your action:",
            options=["合作 🤝", "背叛 ⚔️"],
            index=0,
            horizontal=True,
            label_visibility="collapsed"
        )
        # 反查对应的 Action（后台逻辑不变）
        reverse_map = {v: k for k, v in action_labels.items()}
        choice = reverse_map[choice_label]

        # 大按钮执行本轮
        if st.button("▶ 进行这次决策", type="primary", use_container_width=False):
            user.next_action = Action(choice)
            info = sim.step(overrides={"USER": user.next_action})
            user.next_action = None

            # 弹出结果
            opp_played, my_a, opp_a, my_pay = extract_user_outcome(info)
            if opp_played is not None:
                def translate_action(a):
                    if a in ["C", "c"]:
                        return "合作"
                    elif a in ["D", "d"]:
                        return "背叛"
                    return str(a)

                opp_a_cn = translate_action(opp_a)
                my_a_cn = translate_action(my_a)

                # === 构建中文提示 ===
                flash = f"对手 **{cn(opp_played)}** 选择了 **{opp_a_cn}**；你选择了 **{my_a_cn}** → 本轮获得 **{my_pay:.2f} 分** ⚔️"
                try:
                    st.toast(flash)
                except Exception:
                    st.success(flash)
                st.session_state.last_flash = flash

            # 本轮结束：清空预览，下轮再预览；并立刻重绘以更新 Round/统计条
            st.session_state.preview_pairs = None
            st.rerun()

    with right:
        st.subheader("对手历史决策数据")
        if opp_name is None:
            st.info("还没有对手.")
        else:
            c_pct, d_pct = opponent_cd_percent_global(sim, opp_name)
            render_cd_bar(c_pct, d_pct, opp_name)

            # 👇 新增：显示这个人上一次对我做了什么
            opp_agent = get_agent_by_name(sim, opp_name)
            render_last_action(user, opp_agent)

    # Leaderboard
    st.markdown("---")
    st.markdown(
        f"<h3 style='font-size:26px; font-weight:800; color:#1e293b;'>"
        f"第 <span style='color:#2563eb;'>{sim.round}</span> 天 · 争霸排行榜"
        f"<span style='font-size:16px; color:#6b7280;'>（平均收益）</span>"
        f"</h3>",
        unsafe_allow_html=True
    )

    df = pd.DataFrame(sim.summary(), columns=["Agent", "Total", "Avg/Round"])

    # 按平均收益排序，保留所有
    df_plot = df.sort_values("Avg/Round", ascending=False).copy()

    # ✅ 增加中文显示列
    df_plot["Display"] = df_plot["Agent"].apply(lambda x: AGENT_NAME_CN.get(x, x))

    # 确定显示顺序
    x_order = df_plot["Display"].tolist()

    # === 主图：彩色柱状图 ===
    bars = (
        alt.Chart(df_plot)
        .mark_bar()
        .encode(
            x=alt.X(
                "Display:N",
                sort=x_order,
                title=None,
                axis=alt.Axis(labelAngle=0, labelFontSize=11, labelLimit=200)
            ),
            y=alt.Y("Avg/Round:Q", title="平均收益"),
            color=alt.Color(
                "Display:N",
                legend=alt.Legend(
                    title="",
                    orient="top",
                    columns=10,
                    labelFontSize=15,  # ✅ 图例字体更大
                    titleFontSize=20,  # ✅ 图例标题也大一点
                    symbolSize=90,  # ✅ 图例色块更明显
                ),
                scale=alt.Scale(scheme="category20")
            ),
            tooltip=[
                alt.Tooltip("Display:N", title="人物名"),
                alt.Tooltip("Agent:N", title="策略代码"),
                alt.Tooltip("Avg/Round:Q", format=".3f", title="平均收益"),
                alt.Tooltip("Total:Q", format=".1f", title="总收益")
            ],
        )
        .properties(height=420, width="container")
    )

    # === 高亮 USER（诸葛亮） ===
    user_layer = (
        alt.Chart(df_plot[df_plot["Agent"] == "USER"])
        .mark_bar(color="#f4b6c2", stroke="black", strokeWidth=3)
        .encode(
            x=alt.X("Display:N", sort=x_order),
            y=alt.Y("Avg/Round:Q")
        )
    )

    # === 在顶部标注数值 ===
    labels = (
        bars.mark_text(
            align="center",
            baseline="bottom",
            dy=-3,
            fontSize=11
        ).encode(text=alt.Text("Avg/Round:Q", format=".2f"))
    )

    st.altair_chart(bars + user_layer + labels, use_container_width=True)


    # 操作思路简述（尽量精炼）
    STRATEGY_BRIEF = {
        "TFT": "先合作；之后每轮复制对手上一次的选择，合作则合作，背叛则背叛。",
        "gTFT0.15": "先合作；若对手背叛，会以“偶尔原谅、重建合作”为原则，再回到以牙还牙的节奏。",
        "SG3": "先合作；一旦遭背叛，进入一段固定时长的惩罚期，惩罚结束后主动恢复合作。",
        "ALT": "从固定一方开始；随后合作与背叛交替出现，按既定节奏反复切换。",
        "R50": "从合作或背叛中做随机抉择；不记忆历史，始终保持不确定性。",
        "sWSLS20": "先选一种动作；若上一轮结果理想则保持不变，否则切换动作；过程中允许少量随机扰动。",
        "Joss10": "以合作为主；即便相互合作，也会不时加入突袭式背叛，用以打乱对手节奏。",
        "M1": "根据上一轮的双方组合状态决定当前动作；不同状态对应不同的合作倾向。",
        "Gradual": "先合作；首次被背叛时用轻度惩罚，再犯则加重惩罚；对方回归合作后，逐步减轻直至恢复合作。",
        "AC": "从头到尾保持合作；不因对手背叛而改变策略。",
        "AD": "从头到尾保持背叛；不因对手合作而改变策略。",
        "GRIM": "先合作；一旦遭遇背叛，立刻转为永久背叛，不再恢复。",
        "PROB": "开局以试探为主；若发现对方软弱则持续剥削，若发现强硬则转向更稳妥的应对。",
        "sTFT": "先以试探为主；随后进入以牙还牙的节奏，对手合作就合作、背叛就背叛。",
        "WSLS": "先选一种动作；若上一轮结果理想则继续，若不理想则在合作与背叛之间切换。",
        "TF2T": "先合作；容忍单次背叛不还手；若对手连续背叛，才开始报复；对方回归合作后再恢复合作。",
        "CTFT": "先合作；在存在噪声或误会时，优先尝试修复合作；确认对手持续背叛后再进入报复节奏。",
        "Tester": "开局主动试探（偏强硬）；若对手强硬则迅速收敛、转向合作框架；若对手软弱则维持压制。",
        "Majority": "先参考群体或历史多数；随后持续跟随“占多数的做法”，在多数立场变化时同步调整。",
    }

    # 用你现有的 AGENT_NAME_CN 生成展示表（排除 USER）
    rows = []
    for code, cn_name in AGENT_NAME_CN.items():
        if code == "USER":
            continue
        rows.append({
            "中文角色": cn_name,
            "操作思路（简述）": STRATEGY_BRIEF.get(code, "（待补充）")
        })

    df_brief = pd.DataFrame(rows)

    # 按中文角色排序
    df_brief = df_brief.sort_values("中文角色").reset_index(drop=True)

    # ✅ 添加编号列，从 1 开始
    df_brief.index = range(1, len(df_brief) + 1)
    df_brief.index.name = "序号"

    st.markdown("<hr style='border: 1px solid #e5e7eb; margin: 1rem 0;'>", unsafe_allow_html=True)
    with st.expander("📜 角色策略速览（点击展开）", expanded=False):
        st.dataframe(
            df_brief,
            use_container_width=True,
            height=360
        )

    # ===== 下面是新增的可视化 & 下载功能 =====

    # 1) 先把合作率算出来：从 sim.action_counts 里取
    coop_rows = []
    action_counts = getattr(sim, "action_counts", {})
    for agent_name in df_plot["Agent"]:
        counts = action_counts.get(agent_name, {})
        c = counts.get("C", 0)
        d = counts.get("D", 0)
        tot = c + d
        if tot > 0:
            coop_rate = 100.0 * c / tot  # 转百分比
        else:
            coop_rate = None
        coop_rows.append(coop_rate)

    df_plot["CoopRate"] = coop_rows  # 新增一列：合作率(%)

    # 2) 画散点图：x=合作率, y=平均收益, 颜色=角色
    scatter = (
        alt.Chart(df_plot)
        .mark_circle(size=140)
        .encode(
            x=alt.X("CoopRate:Q", title="合作率 (%)"),
            y=alt.Y("Avg/Round:Q", title="平均收益"),
            color=alt.Color(
                "Display:N",
                legend=alt.Legend(
                    title="",
                    orient="top",
                    direction="horizontal",  # ✅ 水平排列，可自动换行形成两行
                    columns=10,  # ✅ 设大一些，自动两行显示
                    labelFontSize=15,  # ✅ 图例字体大
                    titleFontSize=0,  # ✅ 标题大
                    symbolSize=90,  # ✅ 色块明显
                    padding=5,  # ✅ 图例整体留白
                ),
                scale=alt.Scale(scheme="category20")  # ✅ 20种颜色方案
            ),
            tooltip=[
                alt.Tooltip("Display:N", title="角色"),
                alt.Tooltip("Agent:N", title="策略代码"),
                alt.Tooltip("CoopRate:Q", format=".1f", title="合作率(%)"),
                alt.Tooltip("Avg/Round:Q", format=".3f", title="平均收益"),
            ],
        )
        .properties(
            title=alt.TitleParams(
                text="合作率 vs 平均收益",  # ✅ 图标题
                fontSize=20,  # ✅ 标题更大
                fontWeight="bold",  # ✅ 加粗
                anchor="middle",  # ✅ 居中显示
                dy=20  # ✅ 稍微上移一点，视觉更舒服
            ),
            width=600,  # ✅ 方形宽
            height=600  # ✅ 方形高
        )
    )

    st.altair_chart(scatter, use_container_width=True)


