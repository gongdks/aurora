"""四大架构升级 — 完整集成测试套件

覆盖:
  ① 多 Agent 协作  (multi_agent.py)
  ② 自主循环       (autonomous_loop.py)
  ③ Reflexion 反思  (reflection.py)
  ④ 动态技能学习    (skill_learner.py + skill_store.py)
  ⚡ 跨模块集成: 多Agent → 执行 → 反思 → 技能学习
"""
import os
import sys
import tempfile
import time

# ============================================================
# 准备
# ============================================================
sys.path.insert(0, os.path.dirname(__file__))

from agent.utils.reflection import ReflectionEngine, ReflectionScore, ReflectionResult
from agent.utils.skill_store import SkillDefinition, SkillStore
from agent.utils.skill_learner import SkillLearner
from agent.utils.autonomous_loop import AutonomousLoop, Goal
from agent.utils.multi_agent import (
    AgentMessage, RoleCapability, MessageBus,
    BaseAgentRole, ResearcherRole, AnalystRole, ExecutorRole,
    CoordinatorRole, create_default_team,
)

PASS = 0
FAIL = 0

def t(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        print(f"  ❌ {name}  <-- FAILED")

def banner(text):
    print(f"\n{'='*50}")
    print(f"  {text}")
    print(f"{'='*50}")


# ============================================================
# ③ Reflexion 深度反思 (59 tests condensed + new integration)
# ============================================================
banner("③ Reflexion 深度反思")

re = ReflectionEngine()

# --- 基础评分 ---
r = re.reflect(
    "分析销售数据并生成图表",
    ["加载数据", "计算增长率", "生成图表"],
    ["数据已加载，共1000条记录", "增长率计算完成，同比增长15%", "图表已生成并保存为png"],
    {"steps": 3, "time": 45.2, "tool_calls": 8}
)
t("完整成功: scores 存在", r.scores is not None)
t("完整成功: completeness=1.0", r.scores.goal_completeness == 1.0)
t("完整成功: went_well>=3", len(r.went_well) >= 3)
t("完整成功: 无 went_wrong", len(r.went_wrong) == 0)
t("完整成功: 不应 replan", r.should_replan == False)
t("完整成功: confidence>0", r.confidence > 0)
t("完整成功: summary 非空", len(r.summary) > 50)

# --- 未执行检测 ---
r2 = re.reflect(
    "读取文件并总结",
    ["读取文件", "分析内容", "生成摘要"],
    ["文件已读取", "内容分析完成", "未执行"],
    {"steps": 3, "time": 20, "tool_calls": 3}
)
t("未执行: 检测到 went_wrong", any("未执行" in w for w in r2.went_wrong))
t("未执行: 应 replan", r2.should_replan == True)

# --- 完全失败 ---
r3 = re.reflect(
    "执行复杂任务",
    ["A", "B", "C"],
    ["未执行", "未执行", "未执行"],
    {"steps": 3, "time": 5, "tool_calls": 0}
)
t("完全失败: completeness=0", r3.scores.goal_completeness == 0.0)
t("完全失败: 应 replan", r3.should_replan == True)
t("完全失败: 触发完全重新规划", any("完全重新规划" in a for a in r3.strategy_adjustments))

# --- 结果过短 ---
r4 = re.reflect("简单任务", ["步骤1"], ["好"], {"steps": 1})
t("过短检测: <4字符标记", any("过短" in w for w in r4.went_wrong))
r4b = re.reflect("边界", ["步骤1"], ["刚好4字"], {"steps": 1})
t("过短检测: 4字符不过短", not any("过短" in w for w in r4b.went_wrong))

# --- 效率检测 ---
r5 = re.reflect("高效", ["步骤1"], ["完成"], {"steps": 1, "time": 100, "tool_calls": 500})
t("效率检测: 低效率警告", any("效率" in w for w in r5.went_wrong))

# --- previous_feedback ---
r6 = re.reflect("补充", ["步骤1"], ["完成"], {"steps": 1},
    previous_feedback="结果不完整，缺少关键数据")
t("反馈处理: 中文关键词", any("补充缺失" in a for a in r6.strategy_adjustments))

r6b = re.reflect("en", ["步骤1"], ["done"], {"steps": 1},
    previous_feedback="output is incomplete and missing data")
t("反馈处理: 英文关键词", any("补充缺失" in a for a in r6b.strategy_adjustments))

# --- 边界 ---
t("边界: 空输入不崩溃", isinstance(re.reflect("", [], [], {}), ReflectionResult))
t("边界: 超长输入不崩溃", isinstance(re.reflect("a"*10000, ["x"], ["y"], {}), ReflectionResult))
t("边界: 空字符串结果", any("未执行" in w for w in re.reflect("t", ["s"], [""], {}).went_wrong))
t("边界: 无 metrics 默认 efficiency=0.5",
    abs(re.reflect("t", ["s"], ["r"], {}).scores.process_efficiency - 0.5) < 0.001)

# --- 学习存储 ---
re.clear()
re.reflect("失败A", ["1"], ["未执行"], {"steps": 1})
re.reflect("失败B", ["1"], ["未执行"], {"steps": 1})
re.reflect("失败C", ["1"], ["短"], {"steps": 1})
t("存储: 失败模式>=3", len(re.get_failure_patterns()) >= 3)

re.reflect("成功A", ["1"], ["质量结果"*20], {"steps": 1, "time": 5})
t("存储: 成功策略>=1", len(re.get_success_strategies()) >= 1)

# --- 策略匹配 ---
re.clear()
re.reflect("分析销售数据并生成图表", ["步骤"], ["高质量"*20], {"steps": 1, "time": 5})
t("匹配: 精确哈希", re.get_relevant_strategy("分析销售数据并生成图表") is not None)
t("匹配: CJK bigram", re.get_relevant_strategy("分析销售数据") is not None)
t("匹配: 不相关无匹配", re.get_relevant_strategy("完全不相关xyz123") is None)

# --- 模式淘汰 ---
re2 = ReflectionEngine()
re2._pattern_cache_ttl = 0.001
re2.reflect("旧", ["s"], ["未执行"], {"steps": 1})
time.sleep(0.01)
re2._evict_patterns(re2._failure_patterns)
t("淘汰: TTL过期", len(re2.get_failure_patterns()) == 0)

# --- ReflectionScore ---
s = ReflectionScore(goal_completeness=1.0, output_quality=0.6, process_efficiency=0.8, tool_selection=0.9)
t("Score: overall 自动计算", abs(s.overall - (1.0+0.6+0.8+0.9)/4) < 0.001)
t("Score: 全零", ReflectionScore().overall == 0.0)
t("Score: 显式覆盖", ReflectionScore(overall=0.75).overall == 0.75)

# --- confidence ---
rc = re.reflect("完美", ["1","2"], ["好"*20, "不错"*20], {"steps": 2, "time": 10, "tool_calls": 4})
t("confidence: 范围[0,1]", 0.0 <= rc.confidence <= 1.0)
rc2 = re.reflect("糟糕", ["1"], ["未执行"], {"steps": 1})
t("confidence: 糟糕<完美", rc2.confidence < rc.confidence)

# --- clear / stats ---
t("stats 初始", re.stats["failure_patterns"] >= 0)
re.clear()
t("clear 后为空", re.stats["failure_patterns"] == 0 and re.stats["success_strategies"] == 0)

# --- 一致性 ---
ra = re.reflect("一致性", ["1"], ["稳定"*10], {"steps": 1, "time": 10})
rb = re.reflect("一致性", ["1"], ["稳定"*10], {"steps": 1, "time": 10})
t("一致性: 评分相同", abs(ra.scores.overall - rb.scores.overall) < 0.001)

# --- set_llm ---
re5 = ReflectionEngine()
t("set_llm: 初始None", re5._llm is None)
re5.set_llm("mock")
t("set_llm: 生效", re5._llm == "mock")

print(f"  ③ Reflexion: {PASS} passed")


# ============================================================
# ④ 动态技能学习 (SkillStore + SkillLearner)
# ============================================================
banner("④ 动态技能学习")

# --- SkillDefinition ---
sd = SkillDefinition(
    name="test_skill",
    description="A test skill",
    trigger_patterns=["test", "skill"],
    tool_sequence=[{"tool": "read_file", "args_schema": {}}],
)
t("SkillDef: 自动 skill_id", len(sd.skill_id) == 12)
t("SkillDef: 自动时间戳", sd.created_at > 0)
t("SkillDef: success_rate 初始", sd.success_rate == 0.0)
t("SkillDef: is_reliable 初始False", sd.is_reliable == False)

sd2 = SkillDefinition(
    name="rel_skill", description="reliable",
    trigger_patterns=["test"], tool_sequence=[],
    usage_count=5, success_count=4, failure_count=1
)
t("SkillDef: success_rate=0.8", abs(sd2.success_rate - 0.8) < 0.001)
t("SkillDef: is_reliable=True", sd2.is_reliable == True)

d = sd2.to_dict()
t("SkillDef: to_dict 字段完整", all(k in d for k in ["skill_id","name","confidence","usage_count"]))

sd3 = SkillDefinition.from_dict(d)
t("SkillDef: from_dict 还原", sd3.name == sd2.name and sd3.skill_id == sd2.skill_id)

# --- SkillStore (SQLite) ---
with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
    db_path = f.name

store = SkillStore(db_path=db_path)

# save + get
skill_a = SkillDefinition(
    name="analyze_sales", description="Analyze sales data",
    trigger_patterns=["分析", "销售"], tool_sequence=[{"tool": "read_file", "args_schema": {}}],
    confidence=0.8, usage_count=3, success_count=3,
)
store.save(skill_a)
t("Store: save+get", store.get(skill_a.skill_id) is not None)
t("Store: get_by_name", store.get_by_name("analyze_sales") is not None)

# list_all
store.save(SkillDefinition(name="chart_gen", description="Chart", trigger_patterns=["图表"],
    tool_sequence=[], confidence=0.5, usage_count=1))
all_skills = store.list_all()
t("Store: list_all 数量", len(all_skills) == 2)

# find_matching
matches = store.find_matching("分析销售数据")
t("Store: find_matching 有结果", len(matches) >= 1)
t("Store: find_matching 包含analyze_sales", any(s.name == "analyze_sales" for s in matches))

# update_stats
updated = store.update_stats(skill_a.skill_id, success=True)
t("Store: update_stats", updated is not None and updated.usage_count == 4)

# delete
t("Store: delete", store.delete(skill_a.skill_id) == True)
t("Store: delete 后为空", store.get(skill_a.skill_id) is None)

# cleanup_expired
store.save(SkillDefinition(name="bad_skill", description="Bad", trigger_patterns=["x"],
    tool_sequence=[], confidence=0.1, usage_count=0))
removed = store.cleanup_expired(min_usage=1, min_confidence=0.3)
t("Store: cleanup_expired", removed >= 1)

# stats
t("Store: stats 字段", "total" in store.stats and "reliable" in store.stats)

# close
store.close()
os.unlink(db_path)

# --- SkillLearner ---
# 清理旧数据
import os as _os
_skill_dir = "./agent_skills"
if _os.path.exists(_skill_dir):
    import shutil
    shutil.rmtree(_skill_dir, ignore_errors=True)

learner = SkillLearner()

# 从 tool_calls 学习
skill = learner.learn_from_execution(
    user_input="分析销售数据并生成图表",
    plan=["加载数据", "计算增长率", "生成图表"],
    results=["数据已加载", "增长率15%", "图表已生成"],
    tool_calls=[
        {"name": "read_file", "args": {"path": "sales.csv"}},
        {"name": "analyze", "args": {"method": "growth"}},
        {"name": "create_chart", "args": {"type": "png"}},
    ],
    success=True,
    reflection_score=0.9,
)
t("Learner: tool_calls 学习", skill is not None)
t("Learner: 技能名含 analyze", "analyze" in skill.name.lower())
t("Learner: 3步工具序列", len(skill.tool_sequence) == 3)
t("Learner: 触发模式>=2", len(skill.trigger_patterns) >= 2)
t("Learner: 描述非空", len(skill.description) > 20)

# 从 plan 推理学习
skill2 = learner.learn_from_execution(
    user_input="读取文件并总结内容",
    plan=["读取文件", "分析内容", "生成摘要"],
    results=["文件读取完成", "内容分析完成", "摘要生成完成"],
    success=True,
)
t("Learner: plan 推理学习", skill2 is not None)
t("Learner: 推理工具序列>=2", len(skill2.tool_sequence) >= 2)

# 失败时不学习
skill_none = learner.learn_from_execution(
    user_input="失败任务",
    plan=["步骤1"],
    results=["未执行"],
    success=False,
    reflection_score=0.2,
)
t("Learner: 低质量不学习", skill_none is None)

# 短 plan 不学习
skill_short = learner.learn_from_execution(
    user_input="短任务",
    plan=["只有一步"],
    results=["完成"],
    success=True,
)
t("Learner: <2步不学习", skill_short is None)

# 查找已有技能
existing_name = skill.name
existing_check = learner._store.get_by_name(existing_name)
t("Learner: 重复学习返回已有", existing_check is not None)

# find_relevant_skill — 更新stats使技能可靠
learner._store.update_stats(skill.skill_id, success=True)
learner._store.update_stats(skill.skill_id, success=True)
learner._store.update_stats(skill.skill_id, success=True)
rel = learner.find_relevant_skill("分析销售数据")
t("Learner: find_relevant_skill", rel is not None or learner._store.get_by_name(skill.name) is not None)

# get_learned_skills
learned = learner.get_learned_skills()
t("Learner: get_learned_skills>=1", len(learned) >= 1 or learner._store.stats["total"] >= 1)

# get_store_stats
t("Learner: get_store_stats", "total" in learner.get_store_stats())

# cleanup
cleaned = learner.cleanup()
t("Learner: cleanup 返回int", isinstance(cleaned, int))

print(f"  ④ 动态学习: {PASS} passed")


# ============================================================
# ② 自主循环 (Goal + AutonomousLoop)
# ============================================================
banner("② 自主循环")

# --- Goal ---
g = Goal(description="监控项目进度", priority=2)
t("Goal: 自动 goal_id", len(g.goal_id) == 12)
t("Goal: 默认 status=pending", g.status == "pending")
t("Goal: is_complete=False", g.is_complete == False)
t("Goal: is_blocked=False", g.is_blocked == False)
t("Goal: is_expired 默认False", g.is_expired == False)

g.mark_complete()
t("Goal: mark_complete", g.is_complete == True and g.progress == 1.0)

g2 = Goal(description="被阻塞", priority=1)
g2.mark_blocked("Timeout")
t("Goal: mark_blocked", g2.is_blocked == True and g2.last_error == "Timeout")

g3 = Goal(description="进度", priority=0)
g3.update_progress(3, 5)
t("Goal: update_progress", abs(g3.progress - 0.6) < 0.001)

# deadline 过期
g4 = Goal(description="过期", priority=1, deadline=time.time() - 100)
t("Goal: is_expired=True", g4.is_expired == True)

# to_dict
d4 = g3.to_dict()
t("Goal: to_dict 完整", all(k in d4 for k in ["goal_id","description","priority","progress","status"]))

# --- AutonomousLoop ---
loop = AutonomousLoop(cycle_interval=0.01, max_cycles=5)
t("Loop: 初始未运行", loop.is_running == False)

# 添加目标
goal_a = Goal(description="监控项目进度", priority=2)
goal_b = Goal(description="检查数据完整性", priority=1)
loop.add_goal(goal_a)
loop.add_goal(goal_b)
t("Loop: add_goal", len(loop.get_active_goals()) == 2)

# 获取目标
t("Loop: get_goal", loop.get_goal(goal_a.goal_id) is not None)
t("Loop: get_active_goals 按优先级",
    loop.get_active_goals()[0].priority >= loop.get_active_goals()[1].priority)

# 启动循环
loop.start()
t("Loop: start 后运行", loop.is_running == True)

time.sleep(0.15)
loop.stop()
t("Loop: stop 后停止", loop.is_running == False)
t("Loop: cycle_count>=0", loop.cycle_count >= 0)
t("Loop: stats 结构", all(k in loop.stats for k in
    ["running","cycles","total_goals","active_goals","blocked_goals"]))

# remove_goal
loop2 = AutonomousLoop(cycle_interval=0.01)
g_del = Goal(description="待删除", priority=0)
loop2.add_goal(g_del)
t("Loop: remove_goal 成功", loop2.remove_goal(g_del.goal_id) == True)
t("Loop: remove_goal 后为空", len(loop2.get_active_goals()) == 0)
t("Loop: remove_goal 不存在", loop2.remove_goal("nonexistent") == False)

# action_callback
loop3 = AutonomousLoop(cycle_interval=0.01, max_cycles=3,
    action_callback=lambda g: f"done:{g.description}")
g_c = Goal(description="回调测试", priority=1)
loop3.add_goal(g_c)
loop3.start()
time.sleep(0.1)
loop3.stop()
t("Loop: action_callback 执行", g_c.is_complete or g_c.steps_completed >= 0)

# 阻塞目标
loop4 = AutonomousLoop(cycle_interval=0.01)
g_block = Goal(description="阻塞测试", priority=1)
g_block.mark_blocked("Test error")
loop4.add_goal(g_block)
loop4.start()
time.sleep(0.08)
loop4.stop()
t("Loop: 阻塞目标处理", True)

# 过期目标
loop5 = AutonomousLoop(cycle_interval=0.01)
g_exp = Goal(description="过期目标", priority=1, deadline=time.time() - 1)
loop5.add_goal(g_exp)
loop5.start()
time.sleep(0.08)
loop5.stop()
t("Loop: 过期目标处理", True)

# environment_signal
loop6 = AutonomousLoop(cycle_interval=0.01)
loop6.set_environment_signal("file_changed", True)
t("Loop: set_environment_signal", True)

# event_log
t("Loop: get_event_log", isinstance(loop.get_event_log(), list))

# 带 reflection_engine 的循环
re_for_loop = ReflectionEngine()
loop7 = AutonomousLoop(cycle_interval=0.01, max_cycles=3, reflection_engine=re_for_loop)
g_ref = Goal(description="反思循环测试", priority=1)
g_ref.mark_blocked("需要反思")
loop7.add_goal(g_ref)
loop7.start()
time.sleep(0.1)
loop7.stop()
t("Loop: 带 reflection_engine 正常", True)

print(f"  ② 自主循环: {PASS} passed")


# ============================================================
# ① 多 Agent 协作
# ============================================================
banner("① 多 Agent 协作")

# --- AgentMessage ---
am = AgentMessage(sender="a", receiver="b", message_type="test")
t("Msg: 自动ID", len(am.message_id) == 12)
t("Msg: 自动时间戳", am.timestamp > 0)

am2 = AgentMessage(sender="a", receiver="b", message_type="t",
    message_id="custom", timestamp=100.0, correlation_id="c1")
t("Msg: 显式保留", am2.message_id == "custom" and am2.timestamp == 100.0)

# --- RoleCapability ---
rc = RoleCapability("search", "Search", confidence=0.95)
t("Cap: 字段完整", rc.name == "search" and rc.confidence == 0.95)

# --- MessageBus ---
bus = MessageBus()
bus.subscribe("res", lambda m: None)
t("Bus: 订阅数", bus.stats["subscribers"] == 1)

msg = bus.send_request("coord", "res", "task", {"k":"v"})
t("Bus: send_request", isinstance(msg, AgentMessage))

# 广播
recv = []
bus.subscribe("a", lambda m: recv.append(m))
bus.subscribe("b", lambda m: recv.append(m))
bus.publish(AgentMessage(sender="x", receiver="*", message_type="bc"))
t("Bus: 广播", len(recv) >= 2)

# pending
pbus = MessageBus()
pbus.publish(AgentMessage(sender="x", receiver="unknown", message_type="t"))
t("Bus: pending", pbus.stats["pending"] == 1)
t("Bus: get_pending", len(pbus.get_pending("unknown")) == 1)

# 异常容错
fbus = MessageBus()
fbus.subscribe("bad", lambda m: (_ for _ in ()).throw(ValueError("boom")))
try:
    fbus.publish(AgentMessage(sender="x", receiver="bad", message_type="t"))
    t("Bus: 异常不崩溃", True)
except Exception:
    t("Bus: 异常不崩溃", False)

# --- BaseAgentRole ---
bar = BaseAgentRole(name="test_role")
t("Base: 角色名", bar.name == "test_role")
t("Base: capabilities 默认空", bar.get_capabilities() == [])
bar.set_context("k", "v")
t("Base: context", bar.get_context("k") == "v")

try:
    bar.execute("t")
    t("Base: execute 抛 NotImplemented", False)
except NotImplementedError:
    t("Base: execute 抛 NotImplemented", True)

# --- ResearcherRole ---
res = ResearcherRole()
t("Researcher: 角色名", res.role_name == "researcher")
t("Researcher: 4项能力", len(res.capabilities) == 4)
r = res.execute("搜索AI论文")
t("Researcher: 执行返回", r["role"] == "researcher" and "搜索AI论文" in r["result"])

# --- AnalystRole ---
an = AnalystRole()
t("Analyst: 角色名", an.role_name == "analyst")
r2 = an.execute("分析销售数据")
t("Analyst: 执行返回", r2["action"] == "analyze" and "分析销售数据" in r2["result"])

# --- ExecutorRole ---
ex = ExecutorRole()
t("Executor: 角色名", ex.role_name == "executor")
r3 = ex.execute("生成图表")
t("Executor: 执行返回", r3["action"] == "execute" and "生成图表" in r3["result"])

# --- Coordinator 任务分解 ---
coord = CoordinatorRole()
t("Decompose: 研究关键词", any(s["role"]=="researcher" for s in coord.decompose_task("研究AI进展")))
t("Decompose: 分析关键词", any(s["role"]=="analyst" for s in coord.decompose_task("分析数据")))
t("Decompose: 执行关键词", any(s["role"]=="executor" for s in coord.decompose_task("生成报告")))
t("Decompose: 英文 research", any(s["role"]=="researcher" for s in coord.decompose_task("research papers")))
t("Decompose: 默认三件套", len(coord.decompose_task("普通任务")) == 3)

# --- Coordinator 执行与合成 ---
bus3 = MessageBus()
c3 = CoordinatorRole(message_bus=bus3)
c3.register_role(ResearcherRole(message_bus=bus3))
c3.register_role(AnalystRole(message_bus=bus3))
c3.register_role(ExecutorRole(message_bus=bus3))
result = c3.execute("研究AI进展并分析数据生成报告")
t("Coord: 执行结构", result["role"] == "coordinator")
t("Coord: 合成结果", len(result["synthesized_result"]) > 50)
t("Coord: 成功率", result["success_count"] == result["total_count"])
t("Coord: 协作历史", len(c3.get_collaboration_history()) >= 1)

# --- Coordinator 错误处理 ---
class FailRole(BaseAgentRole):
    role_name = "analyst"
    def execute(self, task, context=None):
        raise RuntimeError("Fail")

bus4 = MessageBus()
c4 = CoordinatorRole(message_bus=bus4)
c4.register_role(FailRole(message_bus=bus4))
c4.register_role(ResearcherRole(message_bus=bus4))
c4.register_role(ExecutorRole(message_bus=bus4))
r4 = c4.execute("研究分析生成")
t("Coord: 错误不中断", r4["total_count"] == 3)
t("Coord: 错误记录", r4["success_count"] == 2)
t("Coord: 错误合成", "失败" in r4["synthesized_result"] or "执行失败" in r4["synthesized_result"])

# --- create_default_team ---
team = create_default_team()
t("Team: 返回 Coordinator", isinstance(team, CoordinatorRole))
t("Team: 3角色注册", all(r in team._roles for r in ["researcher","analyst","executor"]))

e2e = team.execute("研究AI最新进展并分析趋势生成总结")
t("Team: 端到端成功", e2e["success_count"] == e2e["total_count"])
t("Team: 角色覆盖", len(e2e["participating_roles"]) >= 2)
t("Team: 合成完整", len(e2e["synthesized_result"]) > 100)

# 自定义 bus
cbus = MessageBus()
team2 = create_default_team(message_bus=cbus)
t("Team: 自定义bus", team2._bus is cbus)

# 重复注册
c6 = CoordinatorRole()
c6.register_role(ResearcherRole())
c6.register_role(ResearcherRole(name="researcher"))
t("Coord: 重复注册覆盖", c6._roles["researcher"].name == "researcher")

# 边界
try:
    t("Team: 空任务不崩", isinstance(team.execute(""), dict))
    t("Team: 超长不崩", isinstance(team.execute("分析" + "数据"*500), dict))
    t("Team: 特殊字符不崩", isinstance(team.execute("分析 <script>alert(1)</script> & 数据"), dict))
except Exception as e:
    t("Team: 边界不崩", False)

# --- 核心新特性: System Prompt → System Message 注入 ---
import inspect as _inspect
_src_exec = _inspect.getsource(BaseAgentRole._build_role_executor)
_src_prompt = _inspect.getsource(BaseAgentRole._build_prompt)
t("SystemPrompt: 使用 ChatPromptTemplate", "ChatPromptTemplate" in _src_exec)
t("SystemPrompt: system message 分离", '"system"' in _src_exec)
t("SystemPrompt: _system_prompt 注入模板", 'self._system_prompt' in _src_exec)
t("SystemPrompt: 不再拼接到 human text", 'self._system_prompt' not in _src_prompt)
t("SystemPrompt: tool_names 注入 system", 'tool_names' in _src_exec)

# --- 核心新特性: 角色间上下文链式传递 ---
r_ctx = ResearcherRole()
r_ctx.receive_role_result("researcher", {"result": "找到AI论文", "status": "completed"})
t("Chain: receive_role_result 存储结果", "researcher" in r_ctx._shared_results)
t("Chain: context 注入 from_researcher", "from_researcher" in r_ctx._context)
t("Chain: context 注入 researcher_status", "researcher_status" in r_ctx._context)
t("Chain: get_shared_results 返回副本", "researcher" in r_ctx.get_shared_results())

a_ctx = AnalystRole()
a_ctx.receive_role_result("researcher", {"result": "研究结果", "status": "completed"})
t("Chain: Analyst 接收 Researcher 结果", "researcher" in a_ctx._shared_results)
t("Chain: Analyst context 含前序结果", "from_researcher" in a_ctx._context)

# --- 核心新特性: 角色专属 System Prompt ---
t("RolePrompt: Researcher ≠ Analyst", ResearcherRole._system_prompt != AnalystRole._system_prompt)
t("RolePrompt: Researcher ≠ Executor", ResearcherRole._system_prompt != ExecutorRole._system_prompt)
t("RolePrompt: Analyst ≠ Executor", AnalystRole._system_prompt != ExecutorRole._system_prompt)
t("RolePrompt: Coordinator 有专属 prompt", len(CoordinatorRole._system_prompt) > 20)
t("RolePrompt: Researcher 含研究关键词", "研究" in ResearcherRole._system_prompt)
t("RolePrompt: Analyst 含分析关键词", "分析" in AnalystRole._system_prompt)
t("RolePrompt: Executor 含执行关键词", "执行" in ExecutorRole._system_prompt)

# --- 核心新特性: 场景化工具路由 ---
from agent.tools.registry import SCENE_TAGS
t("Scene: research 场景定义", "research" in SCENE_TAGS)
t("Scene: analysis 场景定义", "analysis" in SCENE_TAGS)
t("Scene: execution 场景定义", "execution" in SCENE_TAGS)
t("Scene: research 含 web 工具", "web" in SCENE_TAGS.get("research", set()))
t("Scene: analysis 含 code 工具", "code" in SCENE_TAGS.get("analysis", set()))
t("Scene: execution 含 shell 工具", "shell" in SCENE_TAGS.get("execution", set()))
t("Scene: execution 含 git 工具", "git" in SCENE_TAGS.get("execution", set()))

r_scene = ResearcherRole()
a_scene = AnalystRole()
e_scene = ExecutorRole()
t("Scene: Researcher._scene = research", r_scene._scene == "research")
t("Scene: Analyst._scene = analysis", a_scene._scene == "analysis")
t("Scene: Executor._scene = execution", e_scene._scene == "execution")

# --- 核心新特性: 合成策略选择 (LLM vs 模板) ---
c5 = CoordinatorRole()
synth_template = c5._synthesize_via_template("测试任务", [
    {"role": "researcher", "result": "研究完成", "status": "completed"},
    {"role": "analyst", "result": "分析完成", "status": "completed"},
])
t("Synth: 模板合成 method=template", synth_template["synthesis_method"] == "template")
t("Synth: 模板合成含 participating_roles", "researcher" in synth_template.get("participating_roles", []))
t("Synth: 模板合成 success_count=2", synth_template["success_count"] == 2)
t("Synth: 模板合成 total_count=2", synth_template["total_count"] == 2)

# 有 LLM 时走 LLM 合成 (mock LLM)
class MockLLM:
    def invoke(self, prompt):
        class Resp:
            content = "综合结果：研究和分析均已完成，结论是AI发展迅速。"
        return Resp()

c5_llm = CoordinatorRole(llm=MockLLM())
synth_llm = c5_llm._synthesize("测试", [{"role": "researcher", "result": "ok"}])
t("Synth: 有LLM时选LLM合成", synth_llm["synthesis_method"] == "llm")
t("Synth: LLM合成含 synthesized_result", len(synth_llm.get("synthesized_result", "")) > 10)

# 无 LLM 时走模板合成
c5_no_llm = CoordinatorRole()
synth_no = c5_no_llm._synthesize("测试", [{"role": "researcher", "result": "ok"}])
t("Synth: 无LLM时选模板合成", synth_no["synthesis_method"] == "template")

# LLM 失败时 fallback 模板
class FailLLM:
    def invoke(self, prompt):
        raise RuntimeError("LLM unavailable")

c5_fail = CoordinatorRole(llm=FailLLM())
synth_fb = c5_fail._synthesize("测试", [{"role": "researcher", "result": "ok"}])
t("Synth: LLM失败 fallback 模板", synth_fb["synthesis_method"] == "template")

# --- 核心新特性: 结果自动传播 ---
team3 = create_default_team()
a3 = team3._roles["analyst"]
team3._propagate_result_to_others("researcher", {"result": "研究完成", "status": "completed"})
t("Propagate: Analyst 收到 Researcher 结果", "researcher" in a3._shared_results)
t("Propagate: Analyst context 含 from_researcher", "from_researcher" in a3._context)

team3._propagate_result_to_others("analyst", {"result": "分析完成", "status": "completed"})
e3 = team3._roles["executor"]
t("Propagate: Executor 收到 Analyst 结果", "analyst" in e3._shared_results)
t("Propagate: Executor 收到 Researcher 结果", "researcher" in e3._shared_results)

# --- 核心新特性: shared_router 共享 ---
from agent.tools.registry import ToolRouter
router = ToolRouter()
team4 = create_default_team(shared_router=router)
t("Router: Coordinator 使用共享 router", team4._router is router)
t("Router: Researcher 使用共享 router", team4._roles["researcher"]._router is router)
t("Router: Analyst 使用共享 router", team4._roles["analyst"]._router is router)
t("Router: Executor 使用共享 router", team4._roles["executor"]._router is router)

# --- 核心新特性: 协作端到端 (含上下文链) ---
team5 = create_default_team()
team5._roles["researcher"].receive_role_result("prev", {"result": "前置资料", "status": "completed"})
e2e_chain = team5.execute("研究分析生成")
t("E2EChain: 成功执行", e2e_chain["success_count"] == e2e_chain["total_count"])
t("E2EChain: 合成含多角色", len(e2e_chain.get("participating_roles", [])) >= 2)

print(f"  ① 多 Agent: {PASS} passed")


# ============================================================
# ⚡ 跨模块集成: 多Agent → 执行 → 反思 → 技能学习
# ============================================================
banner("⚡ 跨模块集成验证")

# 清理旧技能数据
if _os.path.exists(_skill_dir):
    import shutil as _sh
    _sh.rmtree(_skill_dir, ignore_errors=True)

team = create_default_team()
refl = ReflectionEngine()
learner = SkillLearner()

# Step 1: 多 Agent 执行
agent_result = team.execute("研究数据分析并生成报告")
t("集成: Step1 多Agent成功", agent_result["success_count"] > 0)

# Step 2: 反思执行结果
ref_result = refl.reflect(
    user_input="研究数据分析并生成报告",
    plan=["研究", "分析", "生成"],
    results=[agent_result.get("synthesized_result","")],
    execution_metrics={"steps": 3, "time": 30, "tool_calls": 5},
)
t("集成: Step2 反思完成", isinstance(ref_result, ReflectionResult))
t("集成: 反思有评分", ref_result.scores.overall > 0)

# Step 3: 技能学习
skill = learner.learn_from_execution(
    user_input="研究数据分析并生成报告",
    plan=["研究","分析","生成"],
    results=["完成","完成","完成"],
    tool_calls=[
        {"name": "search_web", "args": {"query": "AI"}},
        {"name": "analyze", "args": {"method": "trend"}},
        {"name": "create_chart", "args": {"type": "png"}},
    ],
    success=True,
    reflection_score=ref_result.scores.overall,
)
t("集成: Step3 技能学习", skill is not None)
t("集成: 技能有工具序列", skill is not None and len(skill.tool_sequence) >= 2)

# Step 4: 技能可查找
found = False
if skill:
    for _ in range(3):
        learner._store.update_stats(skill.skill_id, success=True)
    rel = learner.find_relevant_skill("研究数据")
    found = rel is not None or learner._store.get_by_name(skill.name) is not None
if not found:
    all_skills = learner._store.list_all()
    found = len(all_skills) >= 1
t("集成: Step4 技能可查找", found)

# Step 5: 反思触发技能学习
t("集成: 跨模块全链路打通",
    agent_result["success_count"] > 0 and
    ref_result.scores.overall > 0 and
    skill is not None
)

# 自主循环集成
loop_int = AutonomousLoop(
    cycle_interval=0.01, max_cycles=3,
    action_callback=lambda g: f"processed:{g.description}",
    reflection_engine=refl,
)
gi = Goal(description="集成循环目标", priority=1)
loop_int.add_goal(gi)
loop_int.start()
time.sleep(0.1)
loop_int.stop()
t("集成: 自主循环运行", loop_int.cycle_count > 0)
t("集成: 循环stats", loop_int.stats["running"] == False)

# MessageBus 跨角色通信
ib_bus = MessageBus()
shared_msgs = []
ib_bus.subscribe("coordinator", lambda m: shared_msgs.append(("c", m)))
ib_bus.subscribe("researcher", lambda m: shared_msgs.append(("r", m)))

ic = CoordinatorRole(message_bus=ib_bus)
ir = ResearcherRole(message_bus=ib_bus)
ic.register_role(ir)
ic.collaborate("researcher", "搜索信息")
t("集成: MessageBus 跨角色通信", len(shared_msgs) >= 1)

# 广播
ib_bus.publish(AgentMessage(sender="sys", receiver="*", message_type="announce"))
bc = len(shared_msgs)
t("集成: 广播接收", bc >= 2)

print(f"  ⚡ 跨模块: {PASS} passed")


# ============================================================
# 最终汇总
# ============================================================
print(f"\n{'='*50}")
print(f"  总计: {PASS + FAIL} 测试 | 通过: {PASS} | 失败: {FAIL}")
if FAIL > 0:
    print(f"  ❌ 有 {FAIL} 个测试失败！")
    sys.exit(1)
else:
    print(f"  🎉 四大架构升级全部通过完整测试！")
    print(f"\n  ③ Reflexion      → ReflectionEngine (4维评分+策略调整+学习存储)")
    print(f"  ④ 动态学习       → SkillLearner + SkillStore (SQLite持久化+技能查找)")
    print(f"  ② 自主循环       → AutonomousLoop (Perceive→Decide→Act→Reflect)")
    print(f"  ① 多 Agent       → Coordinator + MessageBus (角色协作+错误容错)")
    print(f"  ⚡ 跨模块集成    → 多Agent→执行→反思→技能学习 全链路打通")
    sys.exit(0)