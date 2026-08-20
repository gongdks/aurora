"""Quick test of skill learner with all new features."""
import sys
sys.path.insert(0, '.')
from agent.utils.skill_store import SkillDefinition, SkillStore
from agent.utils.skill_learner import (
    SkillLearner, SkillExecutionEngine, SkillMatcher, _ToolIntrospector
)
import tempfile
import os

_test_dir = tempfile.mkdtemp(prefix='skill_test_')

print('=== Testing SkillExecutionEngine ===')
engine = SkillExecutionEngine()
print(f'Engine created: {engine}')

print('\n=== Testing SkillLearner ===')
learner = SkillLearner()
print('Learner created')

print('\n--- Test 1: learn from tool_calls ---')
skill = learner.learn_from_execution(
    user_input='analyze sales data',
    plan=['calc growth', 'list files', 'calc total'],
    results=['15% growth', 'file1.py', '42'],
    tool_calls=[
        {'name': 'calculator', 'args': {'expression': '6 * 7'}},
        {'name': 'list_dir', 'args': {'path': '.'}},
        {'name': 'calculator', 'args': {'expression': '100 / 4'}},
    ],
    success=True,
    reflection_score=0.9,
)
if skill:
    print(f'  skill.name: {skill.name}')
    print(f'  tool_sequence length: {len(skill.tool_sequence)}')
    print(f'  tool_schemas length: {len(skill.tool_schemas)}')
    print(f'  semantic_vector length: {len(skill.semantic_vector)}')
    for step in skill.tool_sequence:
        print(f'    step: tool={step.get("tool")}, args={step.get("args_schema")}')
else:
    print('  NO SKILL LEARNED!')

print('\n--- Test 2: resolve tool names ---')
print(f'  read_file -> {learner._resolve_tool_name("read_file")}')
print(f'  analyze -> {learner._resolve_tool_name("analyze")}')
print(f'  calculator -> {learner._resolve_tool_name("calculator")}')
print(f'  search_web -> {learner._resolve_tool_name("search_web")}')
print(f'  list_dir -> {learner._resolve_tool_name("list_dir")}')
print(f'  nonexistent_xyz -> {learner._resolve_tool_name("nonexistent_xyz")}')

print('\n--- Test 3: infer from step text ---')
print(f'  "读取文件内容" -> {learner._infer_tool_from_step("读取文件内容")}')
print(f'  "搜索AI最新进展" -> {learner._infer_tool_from_step("搜索AI最新进展")}')
print(f'  "分析代码结构" -> {learner._infer_tool_from_step("分析代码结构")}')
print(f'  "计算增长率" -> {learner._infer_tool_from_step("计算增长率")}')

print('\n--- Test 4: execute skill (with runtime params) ---')
if skill:
    result = engine.execute_skill(skill, path=_test_dir)
    print(f'  overall_success: {result["overall_success"]}')
    print(f'  failure_count: {result["failure_count"]}')
    for s in result['steps']:
        print(f'    step {s["step"]}: tool={s["tool"]}, resolved={s["resolved"]}, status={s["status"]}')
        if s['status'] == 'success':
            print(f'      result: {s.get("result", "")[:120]}')

print('\n--- Test 5: promote_tools with scene tags ---')
if skill:
    ok = learner.promote_tools(skill, engine=engine)
    print(f'  promote_tools returned: {ok}')
    tags = learner._infer_scene_tags(skill)
    print(f'  inferred scene tags: {tags}')

print('\n--- Test 6: engine.resolve_tool ---')
engine2 = SkillExecutionEngine()
for name in ['read_file', 'search_web', 'list_dir', 'analyze', 'calculator', 'nonexistent_tool_xyz']:
    resolved, obj = engine2._resolve_tool(name)
    print(f'  {name} -> resolved={resolved}, obj_found={obj is not None}')

print('\n--- Test 7: validate_skill ---')
if skill:
    validation = engine.validate_skill(skill)
    print(f'  valid: {validation["valid"]}')
    print(f'  total_steps: {validation["total_steps"]}')
    print(f'  resolved: {validation["resolved_count"]}')
    print(f'  unresolved: {validation["unresolved_count"]}')
    if validation['issues']:
        for issue in validation['issues']:
            print(f'    issue: {issue}')

print('\n--- Test 8: execute_multiple ---')
skill2 = learner.learn_from_execution(
    user_input='calculate and list',
    plan=['calc result', 'list files'],
    results=['42', 'file1.py'],
    tool_calls=[
        {'name': 'calculator', 'args': {'expression': '6 * 7'}},
        {'name': 'list_dir', 'args': {'path': _test_dir}},
    ],
    success=True,
    reflection_score=1.0,
)
if skill2:
    print(f'  skill2.name: {skill2.name}')
    multi_results = engine.execute_multiple([skill, skill2], path=_test_dir)
    print(f'  multiple results count: {len(multi_results)}')
    for mr in multi_results:
        print(f'    {mr["skill_name"]}: success={mr["overall_success"]}')

print('\n--- Test 9: SkillMatcher features ---')
matcher = SkillMatcher(learner._store)
print(f'  semantic_available: {matcher.semantic_available}')

# Update stats to make skills reliable
if skill:
    for _ in range(3):
        learner._store.update_stats(skill.skill_id, success=True)
if skill2:
    for _ in range(3):
        learner._store.update_stats(skill2.skill_id, success=True)

# Test hybrid search
hybrid_results = matcher.hybrid_search('analyze sales', top_k=3)
print(f'  hybrid_search results: {len(hybrid_results)}')
for hs, score in hybrid_results:
    print(f'    {hs.name}: score={score:.3f}')

# Test fuzzy match
fuzzy_results = matcher.fuzzy_match('analyze', min_score=0.3)
print(f'  fuzzy_match results: {len(fuzzy_results)}')
for fs, score in fuzzy_results:
    print(f'    {fs.name}: score={score:.3f}')

# Test explain_match
if hybrid_results:
    explanation = matcher.explain_match('analyze sales', hybrid_results[0][0])
    print(f'  explain_match for "{hybrid_results[0][0].name}":')
    for reason in explanation['reasons']:
        print(f'    - {reason}')

print('\n--- Test 10: _ToolIntrospector ---')
introspector = _ToolIntrospector()
params = introspector.get_param_names('text_reader')
print(f'  text_reader params: {params}')
aliases = introspector.get_resolved_param_aliases('text_reader')
print(f'  text_reader aliases: {aliases}')

print('\n--- Test 11: execute_skill_for_query ---')
result = learner.execute_skill_for_query('analyze sales', path=_test_dir)
if result:
    print(f'  Found and executed skill: {result["skill_name"]}')
    print(f'  Overall success: {result["overall_success"]}')
else:
    print('  No matching skill found')

print('\n--- Test 12: get_skill_stats ---')
stats = learner.get_skill_stats()
print(f'  total skills: {stats["total"]}')
print(f'  reliable: {stats["reliable"]}')
print(f'  avg_confidence: {stats["avg_confidence"]:.3f}')

print('\n--- Test 13: list_reliable_skills ---')
reliable = learner.list_reliable_skills()
print(f'  reliable skills count: {len(reliable)}')
for rs in reliable:
    print(f'    {rs.name}: usage={rs.usage_count}, confidence={rs.confidence:.2f}')

print('\n--- Test 14: persist semantic_vector ---')
if skill:
    # Re-fetch from store to verify persistence
    fetched = learner._store.get(skill.skill_id)
    if fetched:
        print(f'  semantic_vector persisted: {len(fetched.semantic_vector)} dimensions')
        print(f'  tool_schemas persisted: {len(fetched.tool_schemas)} schemas')

print('\n--- Test 15: validate_all_skills ---')
all_validations = learner.validate_all_skills()
print(f'  validated skills count: {len(all_validations)}')
for v in all_validations:
    status = 'PASS' if v['valid'] else 'FAIL'
    print(f'    [{status}] {v["skill_name"]}: {v["resolved_count"]}/{v["total_steps"]} resolved')

print('\n=== ALL TESTS DONE ===')