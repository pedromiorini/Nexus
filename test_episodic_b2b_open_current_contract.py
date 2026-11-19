"""C1–C3 e N1–N10 da B2b: o aberto corrente só entra se a chamada pedir.

Critérios aceitos em docs/PONTE_EPISODIC_B2B_CRITERIOS_2026-11-18.md.
Este arquivo só observa. Não altera produção.
"""
import hashlib
import inspect
import subprocess
import unittest
from pathlib import Path

import test_router_episodic_memory_bridge_contract as bridge
from core.cognitivo.episodic_memory import Episode, RealEpisodicMemory

ROOT = Path(__file__).resolve().parent
EXISTING_TEST_SHA256 = {
    "test_action_effects_contract.py": "ec91baf5e3e8b87101e03ec3b3e652142990118622f1301bd66a019f10e25369",
    "test_agent_instruction_versions_contract.py": "10807649e48bb1e065d7e646855c658b36492b13bd728deba9f3206da62e2c4f",
    "test_agent_pr79_composition_contract.py": "4ed7fd83d72072325e952b7f171042d18f73525f580ee4c3fab21fdae57607b2",
    "test_agent_registry_contract.py": "8de446f2221522b164c77e68c1364852d3d1f74847270a93b5b985ffd3c50544",
    "test_agent_replay_contract.py": "9915064907c845052e8d71a2af9c0aa0359d8c4927fa8e30b04fc0f78beb499e",
    "test_analogical_reasoning_contract.py": "507eb52c41ac8b8be5896521819bd60730c9ce6ca81ee270008a739ca52b9cee",
    "test_artefatos_drift_contract.py": "29c7a7a3f2fe2befdb9060658b87763a96e17f0003a50c46b07f7aad4e7ed90d",
    "test_attention_contract.py": "bf88fe8fefa6496e528f8fc9b907e00c679eb335561b4ade70912920f5bc7188",
    "test_attention_signal_contract.py": "6fc8ceafcb58a2ee21eeaec7274062d8bcb7ea417ad00ed154998246736b3699",
    "test_audit_atomic_restore_regression.py": "6dea2aafbefb158fa617e56d1d7d995eebc8e2c1aa27b1064becc5538832aed8",
    "test_capability_benchmark.py": "68e91904f72e4854eb08fcb9a4bded15c3aacb61c5d31b31c203a7f30012cccb",
    "test_causal_eval_contract.py": "1270486c93695efec70351c6c6ed3cbc395be69049be55b4bfd0dd3631a91a3d",
    "test_causal_extraction_contract.py": "4c74b94dab52980246caf9116ebef931cb1ad893673a1467de27c08ebceece49",
    "test_causal_world_model_extraction_contract.py": "84461bd94f2fa50ab40924b437410c870399defbfd791b3cd4f8cda4a94bf34d",
    "test_central_router_contract.py": "069b2afb54835f1acd8ee342ceb1ae92d1ff0931932a47bdfa5f53715af0921c",
    "test_ci_gate_shard_contract.py": "9ffa6a6cddebc57a55da41cdb8fb6bcd27b434f103baddbf9c8bf62955b1a2e1",
    "test_cognitivo_eval_cli_contract.py": "94476fcacf9b27ae1ace2b5b71e89de00c3284fa341429422dfd090baa5eb457",
    "test_cognitivo_mutation_parallel_contract.py": "856106ddfd54430fc4d71ab05c9e6b6e0abfc58c6102c6d66e7bc7a073bd50c5",
    "test_cognitivo_mutation_preflight_edges_contract.py": "834600135a863a1c80e268ccae12e2e005afc7fafb0922535dd3ba75e12bdd9b",
    "test_consensus_application_contract.py": "5936233a5f80ef3d249f15ff6fa52c957ba62f3371ebaab85421ef385ed0c744",
    "test_contextual_memory_contract.py": "8b18cbe6c9fbf9675524a2b0322f48b4176c945c4491872b556572ddb09edd98",
    "test_core_hygiene_contract.py": "3de0b0e1e1217b4fe8ea8093c77f6d829c83d89c088f89cd928c0da13573c4a6",
    "test_correction_claim_audit_contract.py": "c1c79f4c3455e4b392b4146bdb3c720bb276460ff5b621c57e3d161dd960a865",
    "test_counterfactual_contract.py": "afd20e55961724afc31ae67e93d1663fcf9fb5bf6475661ef8b864031d18f764",
    "test_counterfactual_dox_contract.py": "9249ff591787b53394d157348054e9f537849bd75616e8355a28757bc71f4e06",
    "test_counterfactual_eval_contract.py": "f3bf675d00d37268020f8ca6f41cf21b5b70556cd50eb6361eedadc7f716f86c",
    "test_counterfactual_extraction_contract.py": "fb97b1f7cbb11fb49f28d09619979b0005c85dd5f2af17d037930d5a067f1e31",
    "test_counterfactual_gap3_contract.py": "900cb42861f82bde8497d39081b498c6d1521b4eb44667572ea333e419d5f1da",
    "test_counterfactual_gap4_contract.py": "70691d749f5c284dbbd55abdd6acdb2d72aab65638fda7bfb9da385075dc4548",
    "test_counterfactual_planner_contract.py": "89453122449f6139f7438141d21a81f4db89d907c44fbc9ba05492788cb992fe",
    "test_counterfactual_plausibility_contract.py": "ce7b5a894dd19dc535cb9ba2a8f669153b24fcf2c5ce4a1cec2e2d84f6a14203",
    "test_decision_cycle_contract.py": "257984a78b6bad583d77bc64920b1c84f298ca6857d282ac973b01f2619c3395",
    "test_decision_cycle_review_contract.py": "6d58366ce537de2e6475b9775dc52d2e82b70fb6c75668a39b72a14bc5587760",
    "test_deferred_task_properties.py": "0ce69187dc0cb6d761bf3b19a5b35d85b162943cd3da8135d5ac17723abf79ec",
    "test_deferred_task_queue.py": "6aa2f184942d3a1aea4f8ae0ccdfa0a4b69d16e70be453b5763bcac4e9ec0bf5",
    "test_deferred_task_snapshot.py": "4da64c559161ba682822f23d48a93f63757032f9d5184a48a039705ca4e2c43f",
    "test_embedded_assert_audit.py": "81070f33b6035f2c6b4c19afaf99aaff9562f92330c8aaceef5b6ad7b34a74fa",
    "test_embedded_assert_line_contract.py": "0181c330f73e743720df53e325e1c2f33225a6c88c34d64f453ab88acf68e231",
    "test_emotional_memory_weighting_contract.py": "9ef14f7639a1b9b1162f5a1f9c8e037f728be31d13833591d554bc6e00bad92b",
    "test_episodic_future_thinking_contract.py": "0f9b3a42a1f16509a1e205775646797a2b4b75ae10bd9027ed5cf7c16792fc66",
    "test_episodic_like_literal_contract.py": "77ad72999806872d4c893f6a419d0635211a696747428e6145660778994c2c30",
    "test_episodic_memory_concurrency_contract.py": "147d5d3c5b48d8e618c66d62fe0b57c57ca57c283c7c08c67d3ecf50820ad8d7",
    "test_episodic_memory_extraction_contract.py": "19213f817fb1f3f5818dc344494cb980d8cb9a713904fcc951a15f7a19ba101d",
    "test_episodic_memory_search_limits_contract.py": "9688301c1c77472f611fc3ed10f8f321cc9118041871c1a3d3c8c235c1ef7dd3",
    "test_episodic_memory_statistics_contract.py": "939aea93b37e6404c6ecaf1ae1d313f9c8304b52651af4079cc13dc317869fad",
    "test_episodic_retrieval_contract.py": "8adf6c8ff9e2a67031ec96281b9e6c6a110cae4e940d4f998c4e1d32dc41fa41",
    "test_error_correction_contract.py": "c8aa12089e92bfb3d37375a9f473b92a4847263c77cc64e3257a04604857c8ac",
    "test_evidence_omission_contract.py": "12f2f243c5dbcbac041cc9c831e2875c9d01430da1557a4adf757f69ae342629",
    "test_explicit_gap_contract.py": "09b6b9587692b2db94ce4b4c8a414cd7e29b439fee4d8cadd9a53a3019b4e512",
    "test_format_json_contract.py": "028edfe3c271df2ef403598a4c02f85a72a272a455c4ca1013d462d4f72037be",
    "test_futures_ranking_contract.py": "ca89acfc0ac9df8e0e873830625bd740611344ce30397ba6d532eaa449fe2d9e",
    "test_global_workspace_contract.py": "a84a139ef5b9d0903374794a87f0d1fd66bff702cb892aaca8b8e6975daf4bb3",
    "test_grok_individual_acceptance_contract.py": "33e138edafe2313bacaf2590136526b0ea63799d5bf9d08fb154dc73e38aebbf",
    "test_hardware_telemetry_contract.py": "81d15f84086cd0e65c18e1707acedd5dac27c78e41337f45db0b3c7e30c80961",
    "test_immutable_log_contract.py": "c009e34c73a1169aadae4c360b6e45c38bd5a6e98e15d9e80717b28a24574f02",
    "test_implementation_gap_audit.py": "318832b486b731896e0b576068f05124a00d45e52f537ad0e91e617f4e9f3875",
    "test_implementation_gap_raise_class_contract.py": "3f41a6753dea606dd42296d515a20009ab5daf0ea48e6221ee778f34a196c344",
    "test_incremental_consolidate_contract.py": "c0267c2f7070df57c79d690cfc8097e39d718c540b681cb3ec2f120a20c9bd81",
    "test_incremental_learner_contract.py": "5169ff598d6ea3b6dc2101e0e35121d931a021be13ca6524787c84f7a093a3d3",
    "test_integrated_manifest_duplicate_count_contract.py": "2ed613d9658a74b46f22bfd55f0e2f2b2ba4cb012fb722b1dab826b6ae32bed0",
    "test_integrated_suite_summary.py": "a8c73b6b3c2ef60dc0cb36b39c9ad94f7f63b5cb3889884e3078bd7103a2f7a4",
    "test_integrated_summary_duplicates_contract.py": "dca9b2403cbdcd8fb3065daa9832679e47521152bec83258f09e4979aa41d0f5",
    "test_integration_synergy_contract.py": "03a42f67104f468ded21e6046d68ec36c6f8e09fb8225651519ebf5c329cffe4",
    "test_jailbreak_detector_contract.py": "36632fa9caf7b67bad7e14f71bb66646bf2b53f7a7fb3110433d97596f0a0eb1",
    "test_knowledge_graph_contract.py": "d716eee4fef33230fd497e21884b471b0240d49dfd0439668b3bb9dd3bf5ccfd",
    "test_knowledge_graph_endpoint_hops_contract.py": "37f73869676246d88ab19aeda926f6f932513adc221eee36d6f4b2e6ecde758a",
    "test_knowledge_graph_hops_contract.py": "cd067c8184df852f1335923dba347428b867cc00db28c86a48b816b48ac90b89",
    "test_math_calculate_resource_limits.py": "fe01fea8bbeea48398cc0e408c7f8e108d91ded5ade0e8fd39f68a46ec294615",
    "test_mcts_action_contract.py": "08904efd16f2988ccbd620bf74fa1b1f90ab261a656cc4e9f4710fb4e2a40089",
    "test_mcts_compare_veto_contract.py": "59da9a5b334239be510276aae8c39236770512219a57562d5d7724633d90e026",
    "test_mcts_expected_reward_contract.py": "038d8cf15df183cc786871d318c73113483a8df1fe5881eb34644de0a6b0590b",
    "test_mcts_p1b_contract.py": "7d2450fc3a7b266ecc335c5dc0b4899c0a04f41eb69a3f44d94807a6675c3720",
    "test_mcts_p1c_contract.py": "1ed5ae36905d6803d2dfde7c856254dc1d9daa7f63a1ca7ee2841423b2f5b363",
    "test_mcts_p1c_veto_key_errata_contract.py": "b4f860c6f850de3de30047fb5eab0feb0f7c0d4f05eb98e63482e88631ca3fe4",
    "test_mcts_planner_extraction_contract.py": "9d33df751edc22423693050ed529035b6593e5cc9454ed0d400002509fac848c",
    "test_mcts_planner_input_contract.py": "f8e0c10147bae56d7752f89579dc35d44a49f107a1a44141fd5b7e23f326e011",
    "test_mcts_planner_quality_contract.py": "55f21b4b2ddaf6030c282d87c06aa33307912132abf07360a96a9ff989dc4a21",
    "test_mcts_prompt_injection_contract.py": "7c739ebbc9ffccef8d332d23dc3f1f9deac21a4c0f4bc8f1104d6b011d0ba0de",
    "test_mcts_world_effects_contract.py": "196eb28bf8ad0d321112b2f845b25dedd420c2c217cbfa1e26ef26671bb4d982",
    "test_memory_sql_contract.py": "564ddec8e7c013d4981ee8c417c5330ce44542449696d66e6af64234afb0ca17",
    "test_mental_simulation_contract.py": "e5e127c900c5410661317be4ddde085c52d31ff123027ce98cf8d81fa2472470",
    "test_mental_simulation_goal_credit.py": "796975b500a7c180d85a6096658833fb1ad32aa488f960729cbcd293f140b39a",
    "test_multimodal_backend_contract.py": "5d999c8e73e7e897cad9f3d46738b9e7717ab46a0fe580645824769887bf9a19",
    "test_pomdp_belief_filter_contract.py": "9ad31203f03abbfe8b650e56b86272a7f49808cbcc2c6c17bec1b11d498ee095",
    "test_pomdp_benchmark_v2.py": "19dfd8c4edcf990a3df2d8c9598035eb045439e1ab35f170d5d810af2c0d3d6f",
    "test_predictive_coding_extraction_contract.py": "6d9814fd50779986e7448c29ac0c3fa30bfdd0255ec9d68855c3c5b3e7988454",
    "test_predictive_learning_contract.py": "02406c9bec67f5a2d41f004ca2d4e7780fb8dbb4b043d6758f5cba12da412040",
    "test_python_compatibility_gate_contract.py": "389881aa4ec4ad8762644f734c58ebe8644cf72619c288dfd9429eb913a7ba4b",
    "test_reality_audit.py": "10bb2ab8ac861d309bf9bf9a93a4e3b815cf7ae788ca477cd243bb0187985c36",
    "test_reality_audit_cognitivo_contract.py": "c38373b0324832af2d05103264dd05d29c64bc4c9f3bb667b771df673a9861c3",
    "test_recovery_diagnostics_contract.py": "edbc03394723bcd7f4c889e36d20989b3dfe91a73a92f3159b63a93815cfc463",
    "test_red_pr156_veto_names.py": "8afd23a10bc5a3600b57cbbc7e33923d1f0e0bc164b36eb540437e18cdc71fb0",
    "test_red_pr157_pearl.py": "3ebf8ef920f0a8288d92bf20983597c1f0ffa406bfe4ad9815dcbfddc5acfb4c",
    "test_reminiscence_bump_contract.py": "0219481263fb08ecfeac0a729bacf437aa3fa2c3064dc0011b519205f23f0ac4",
    "test_router_budget_swarm_contract.py": "2831d054bb74554db1fabf28870becaf59935e677f07db2f6564d76c1d2f7824",
    "test_router_decision_cycle_bridge_contract.py": "3f0943afdc640b7d267f17b3aaa76d33ac4da0a5402721f6220277e72cc53533",
    "test_router_decision_cycle_bridge_red_contract.py": "29c798684be56339b3474662411685dbb1a54c3cc633ccdded6aadda73b2d290",
    "test_router_episodic_memory_bridge_contract.py": "3d3fabe0ff069ebe9f8143b8b17320156f154b8b44861dc0cc2afa286d5a3893",
    "test_router_episodic_memory_bridge_e1_e6_red_contract.py": "2df0499cd9febc8bde6ed7063b0aafd4781627d728534e8223be2da0cd29832e",
    "test_router_fail_closed_contract.py": "9035b90cf3bd50a68eb029c17d5c517d703751b21a930bf715a1a53a3339062e",
    "test_router_review_addendum_contract.py": "a58a055290d515f4e208289e598e0354ed90668a4fba891887442c1b0f0c6b98",
    "test_scm_counterfactual_contract.py": "383978b45445145a00a33204ccdf05e86c956b09cad878b102e3695bee8fd099",
    "test_scm_counterfactual_review_contract.py": "878dd1a9252e6d9d77c6093f67bd95a35121000e9ec865c94ace2f80853ac0ee",
    "test_scm_counterfactual_second_review_contract.py": "636e1d6fed271bdae4f384c0278dbdec8a63b64137ae99ac184f3df11c205989",
    "test_scm_exact_str_contract.py": "08a58a62dd309cd56e9d500ee5f35ca5b82b6c7f7e1162a76e976b307e8561b8",
    "test_scm_intervention_contract.py": "8e9be97c0a4f1a7ae7a4bb875573249b70b679fbc104b431c95bf2eabd20e943",
    "test_scm_outside_table_contract.py": "35a9c67e95ab1bcebd62a726647b959639b5567f63ba6d37a4693cc8716df52c",
    "test_scm_review_addendum_contract.py": "82ff96b1dff9e7e087b0c5cdbf0328adcb3afbca6f8a1dd8b16448ff557ddf8e",
    "test_scm_schema_contract.py": "dd2430cc6437f34f513063f37244920792bd0ef24da43b9b08f79eb5d9dfd943",
    "test_security_triage.py": "ea6cb6e859fd78d927851502e0c560903bc6cae4d5a14dcc3ca7ec3c5b117001",
    "test_security_triage_line_contract.py": "79c6879a7cf77f091c8c85f08ccb47411be90105ff28caa152a22740b7fe0e8b",
    "test_semantic_networks_contract.py": "e02ec2f214207db9510adab1f3c423abd00a98930ba7bbb27c8cd7ca62cb0e5e",
    "test_semantic_networks_followup_contract.py": "43340224c8c791bf74275e8c82cb97e54ff6aed483764f7219fe053e38990e23",
    "test_semantic_networks_sn3b_sn4b_contract.py": "8bac29e91ffb19a642bc5fc7d5679f8669fadee61a2059548002898b76ead0c9",
    "test_sensorimotor_multimodal_contract.py": "b8ccd5e6e86d187a64d7ac0cab3558c3d0356ba02ef0c757280314894aed291e",
    "test_shadow_clone_reintegration_contract.py": "1872f624b2bd74f3fea86b702c24ec04221916d2627694bfdc4b6a09c93caeb7",
    "test_snapshot_depth_atomicity_contract.py": "9ed1f6c13546e365e351d0aa664a8afe43c0c24d8471a482728a1eda44267f3a",
    "test_state_consistency_audit.py": "7fa0b203cb53e33ea56af84392d3f055e06b9a29a5f0485e04727aafb8bf6e0f",
    "test_state_counter_sentinel_contract.py": "fcb36197b9ae6900f568a0a338f46aecfecd65506a8e20bdbdbc226ba6d1a95b",
    "test_state_manifest.py": "b91bdcaa050f94972f5b29a938852c3b8f2b8d2ffc2a0e75a0182e8b6ca1aa54",
    "test_state_stray_marker_contract.py": "da70f0584fb5c1761a80bcffdda1fdf33dba5ae490d54c23da643944cfab04d6",
    "test_swarm_statistics_contract.py": "714514de77f1b6fdfb4c3ba9a3997e11857add234383bc693689766e6a7fc162",
    "test_system_health_diagnostics_contract.py": "7bdf2e48c955321c2c9ce529e4b69dce6d3b2cff1166f638d4e1c66ba6777967",
    "test_system_health_diagnostics_extraction_contract.py": "840624e73076e9f06b81eb081294b9302657394151bedfe0bab8cc9b690f0a27",
    "test_temporal_facts_contract.py": "7333b5eb57c49b0ccfa80b2beb8dce664200aea061f46a6d5983722ac49279fc",
    "test_temporal_graph_integration_contract.py": "e9f0ebca1ee6b4d6cabb4ea4e355e2b474faac906ec23bf44ae200c370950bd0",
    "test_temporal_head_hash_concurrency_contract.py": "2e6cd6f43a7c12ec763076474e11f755c28aadafee1177cd401da6692b400ef9",
    "test_temporal_reasoning_contract.py": "30108793e5522acc8e272c4e465978498fb6e74d20c59eb26f032847ddaf29d5",
    "test_temporal_snapshot_cut_validation_contract.py": "bd8b3f04a9b1c5a19b48e31b5ac2a931415bd014274ea99da5600f5295cbcca3",
    "test_test_summary.py": "ceebf825b3b9a4b4950f84bf22550523e643a4e0b78fdc1c1556c60a1e8ad3bb",
    "test_timeout_margin_contract.py": "dff35e6b44aedf32410280ebae2504163135e6719378c121d34d7aec0fc92db1",
    "test_tool_cache_hit_contract.py": "76eea335499c35c2c310a280e78a9221d7e721f6b88115daaca6801a9cf0716c",
    "test_tool_cache_key_contract.py": "421a9ac84afd0bad7838d50fb548302bf8c978c6fd337be9044e771bccb3362f",
    "test_tool_cache_snapshot_contract.py": "f4899666cf0cabb94e6f77e5512b1897c1030cd568b313bdacc9ac9845270168",
    "test_tool_cache_store_contract.py": "2674082c21267c0b4351fb52e45e1d27fc85ee3703bdd9d0de7948378553115e",
    "test_tool_definition_contract.py": "1e9b506714d4ed8018c6f7024733726fc66040254313ed4f961a8e29bc910cdb",
    "test_tool_parameter_contract.py": "6bd6621051becc679e1ad7c36916305114c08080bfe5d91328feb665cbfd49d0",
    "test_tool_permission_failclosed_contract.py": "01c79328d0eeee83cdafab1c8620932007b051b16470b5c0841c2a99880bba5a",
    "test_transfer_mapping_contract.py": "b0fa6d0e27c309014a82582998332782ef7434ea61cb890f0aea0c49192c2917",
    "test_veto_key_precedence_contract.py": "00a8b095ce63b389ff17edfabf3fe155d21b54278ffbad1b2ac68c26b7374aba",
    "test_veto_key_retired_mutants_contract.py": "ae485d47a66293ff536926a6fc96099f52846deeef688194c4ac85b0d042d6e7",
    "test_vita_audit_bounds_contract.py": "f4f9ba454408f5a2079500438576f6c3611e9363556286a69b4ea358a5456563",
    "test_vita_audit_concurrency_contract.py": "b5b52a4d8b136589cf75602d55e1479847a9386da212c04db7ede3f827355e35",
    "test_vita_audit_cross_connection_contract.py": "b97db3b16982da35a241335684a19cf39eef1e51e2e7a033d23680cb9669f6f8",
    "test_vita_telemetry_counter_contract.py": "96e4c38cbb814a433f3e5eb5cacdb6356bbe8b6d99be8c3b3a7c22f51908ab96",
    "test_vram_defense_guard.py": "a0330b8c008eb2ca458f83be0d6a412fd8204795d90581211251469287180672",
    "test_working_memory_contract.py": "b11a89af1d2bac48b9fc5e277cdf34bbbc8aa9f830dbb4713df68e13b1043045",
}
LIKE_LITERAL = "test_episodic_like_literal_contract.py"
LIKE_LITERAL_SHA256 = "77ad72999806872d4c893f6a419d0635211a696747428e6145660778994c2c30"
RECALL = "remember the last time"
C1_TITLE = "Q: remember the last time"
THIS_FILE = "test_episodic_b2b_open_current_contract.py"


def _flag_available():
    return "include_current_open" in inspect.signature(RealEpisodicMemory.search_episodes).parameters


def _fields(episodes):
    return [
        (episode.episode_id, episode.title, episode.trigger_query, episode.importance, episode.status)
        for episode in episodes
    ]


class EpisodicB2bOpenCurrentContractTests(unittest.TestCase):
    def _memory(self):
        memory = RealEpisodicMemory(":memory:")
        self.addCleanup(memory.close)
        return memory

    def _open_c1(self, memory):
        return memory.start_episode(C1_TITLE, trigger_query=RECALL, importance=0.7)

    def _closed_plain(self, memory):
        episode_id = memory.start_episode(RECALL)
        memory.close_episode(episode_id)
        return episode_id

    def _like_memory(self):
        """Mesma montagem de EpisodicLikeLiteralContractTests.setUp."""
        memory = self._memory()

        def closed(title, trigger_query=None):
            episode_id = memory.start_episode(title, trigger_query=trigger_query)
            memory.close_episode(episode_id)
            return episode_id

        closed("100%")
        closed("plain")
        closed("a_b")
        closed("axb")
        closed("rate_limit")
        closed("alpha release")
        closed("note", trigger_query="save 100%")
        memory.start_episode("100% open")
        return memory

    def _router(self, memory):
        modules = bridge.harness_modules(memory)
        with bridge.silence():
            router = bridge.CentralRouter(modules, bridge.ethics_guard())
        router.vram_guard = bridge._ObserveGuard()
        return router

    def test_c1_explicit_flag_returns_the_current_open_match(self):
        memory = self._memory()
        episode_id = self._open_c1(memory)
        found = memory.search_episodes(RECALL, limit=3, include_current_open=True)
        self.assertEqual(
            _fields(found),
            [(episode_id, C1_TITLE, RECALL, 0.7, "open")],
        )
        self.assertIsInstance(found[0], Episode)
        self.assertEqual(memory.current_episode_id, episode_id)

    def test_c2_closed_stays_and_current_open_is_appended_when_asked(self):
        memory = self._memory()
        closed_id = self._closed_plain(memory)
        open_id = self._open_c1(memory)
        found = memory.search_episodes(RECALL, limit=3, include_current_open=True)
        self.assertEqual(
            _fields(found),
            [
                (closed_id, RECALL, None, 0.6, "closed"),
                (open_id, C1_TITLE, RECALL, 0.7, "open"),
            ],
        )

    def test_c3_explicit_flag_appends_after_a_full_limit(self):
        memory = self._memory()
        for index in range(3):
            episode_id = memory.start_episode(
                f"remember the last time {index}",
                trigger_query=RECALL,
            )
            memory.close_episode(episode_id)
        self.assertIsNone(memory.current_episode_id)
        prefix = [episode.episode_id for episode in memory.search_episodes(RECALL, limit=3)]
        self.assertEqual(len(prefix), 3)
        open_id = self._open_c1(memory)
        second = [
            episode.episode_id
            for episode in memory.search_episodes(RECALL, limit=3, include_current_open=True)
        ]
        self.assertEqual(second, prefix + [open_id])
        self.assertEqual(second.count(open_id), 1)

    def test_n1_like_literal_file_stays_and_passes(self):
        digest = hashlib.sha256((ROOT / LIKE_LITERAL).read_bytes()).hexdigest()
        self.assertEqual(digest, LIKE_LITERAL_SHA256)
        proc = subprocess.run(
            ["python3", "-m", "unittest", LIKE_LITERAL],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            env={
                "PATH": "/usr/bin:/bin",
                "PYTHONPATH": str(ROOT),
                "NEXUS_AUDIT_DB": "/tmp/nexus_audit_b2b_n1.sqlite3",
                "LC_ALL": "C",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_n2_percent_search_without_flag_keeps_the_open_out(self):
        memory = self._like_memory()
        titles = [episode.title for episode in memory.search_episodes("%")]
        self.assertCountEqual(titles, ["100%", "note"])
        self.assertNotIn("100% open", titles)
        self.assertEqual(
            [episode.title for episode in memory.search_episodes("%", status="open")],
            ["100% open"],
        )
        self.assertEqual(len(memory.search_episodes("%", limit=1)), 1)

    def test_n3_default_search_does_not_append_the_current_open(self):
        memory = self._memory()
        episode_id = self._open_c1(memory)
        self.assertEqual(memory.search_episodes(RECALL, limit=3), [])
        self.assertEqual(memory.current_episode_id, episode_id)

        other = self._memory()
        for index in range(3):
            closed_id = other.start_episode(
                f"remember the last time {index}",
                trigger_query=RECALL,
            )
            other.close_episode(closed_id)
        self.assertIsNone(other.current_episode_id)
        prefix = [episode.episode_id for episode in other.search_episodes(RECALL, limit=3)]
        self.assertEqual(len(prefix), 3)
        self._open_c1(other)
        second = [episode.episode_id for episode in other.search_episodes(RECALL, limit=3)]
        self.assertEqual(second, prefix)

    def test_n4_non_current_open_stays_out_with_or_without_the_flag(self):
        memory = self._memory()
        open_id = memory.start_episode("kernel panic")
        closed_id = memory.start_episode("kernel panic again")
        memory.close_episode(closed_id)
        self.assertIsNone(memory.current_episode_id)
        self.assertNotEqual(open_id, memory.current_episode_id)
        self.assertEqual(
            [episode.episode_id for episode in memory.search_episodes("kernel")],
            [closed_id],
        )
        self.assertEqual(
            [episode.episode_id for episode in memory.search_episodes("kernel", status="open")],
            [open_id],
        )
        if _flag_available():
            flagged = memory.search_episodes("kernel", include_current_open=True)
            self.assertEqual([episode.episode_id for episode in flagged], [closed_id])

    def test_n5_search_without_a_matching_query_does_not_append(self):
        memory = self._memory()
        alpha = memory.start_episode("alpha", trigger_query="q1", importance=0.9)
        beta = memory.start_episode("beta", importance=0.3)
        memory.close_episode(alpha)
        self.assertEqual(memory.current_episode_id, beta)
        self.assertEqual(
            [episode.episode_id for episode in memory.search_episodes(limit=10)],
            [alpha],
        )
        self.assertEqual(
            [episode.episode_id for episode in memory.search_episodes("alp", limit=5)],
            [alpha],
        )
        if _flag_available():
            flagged = memory.search_episodes("alp", limit=5, include_current_open=True)
            self.assertEqual([episode.episode_id for episode in flagged], [alpha])

    def test_n6_limit_zero_and_invalid_input_stay(self):
        memory = self._memory()
        self._open_c1(memory)
        self.assertEqual(memory.search_episodes(RECALL, limit=0), [])
        with self.assertRaises(ValueError):
            memory.search_episodes("")
        with self.assertRaises(TypeError):
            memory.search_episodes(False)
        with self.assertRaises(ValueError):
            memory.search_episodes(RECALL, limit=-1)
        with self.assertRaises(TypeError):
            memory.search_episodes(RECALL, limit=True)
        if _flag_available():
            self.assertEqual(
                memory.search_episodes(RECALL, limit=0, include_current_open=True),
                [],
            )
            with self.assertRaises(ValueError):
                memory.search_episodes("", include_current_open=True)
            with self.assertRaises(TypeError):
                memory.search_episodes(False, include_current_open=True)
            with self.assertRaises(ValueError):
                memory.search_episodes(RECALL, limit=-1, include_current_open=True)
            with self.assertRaises(TypeError):
                memory.search_episodes(RECALL, limit=True, include_current_open=True)

    def test_n7_status_open_does_not_duplicate(self):
        memory = self._memory()
        prior_id = memory.start_episode("remember the last time prior")
        current_id = self._open_c1(memory)
        found = memory.search_episodes(RECALL, status="open", limit=10)
        ids = [episode.episode_id for episode in found]
        self.assertEqual(len(ids), 2)
        self.assertEqual(set(ids), {prior_id, current_id})
        if _flag_available():
            flagged = memory.search_episodes(
                RECALL, status="open", limit=10, include_current_open=True,
            )
            flagged_ids = [episode.episode_id for episode in flagged]
            self.assertEqual(len(flagged_ids), 2)
            self.assertEqual(set(flagged_ids), {prior_id, current_id})
            self.assertEqual(len(flagged_ids), len(set(flagged_ids)))

    def test_n8_bridge_kwargs_stay_query_and_limit(self):
        source = (ROOT / "core/constitutional_brain.py").read_text(encoding="utf-8").splitlines()
        self.assertEqual(
            source[10436],
            "            return module.search_episodes(query=prompt, limit=3, include_current_open=True)",
        )
        router, recorder = bridge.harness_h()
        with bridge.silence():
            first = router.route(bridge.RECALL, {})
            second = router.route(bridge.RECALL, {})
        self.assertEqual(
            recorder.calls,
            [("search_episodes", (), {"query": bridge.RECALL, "limit": 3, "include_current_open": True})],
        )
        self.assertNotIn("status", recorder.calls[0][2])
        self.assertNotIn("episode_id", first)
        self.assertNotIn("episodic", first)
        self.assertIs(second["cached"], True)

        router, recorder = bridge.harness_h()
        context = {"request_type": "query"}
        with bridge.silence():
            router.route(bridge.RECALL, context)
            router.route(bridge.RECALL, context)
        expected = ("search_episodes", (), {"query": bridge.RECALL, "limit": 3, "include_current_open": True})
        self.assertEqual(recorder.calls, [expected, expected])

        memory = RealEpisodicMemory(":memory:")
        self.addCleanup(memory.close)
        episode_id = memory.start_episode(bridge.RECALL)
        memory.close_episode(episode_id)
        seen = []
        original = memory.search_episodes

        def wrapped(*args, **kwargs):
            found = original(*args, **kwargs)
            seen.append((args, dict(kwargs), [episode.episode_id for episode in found]))
            return found

        memory.search_episodes = wrapped
        with bridge.silence():
            result = self._router(memory).route(bridge.RECALL, {})
        self.assertEqual(seen, [((), {"query": bridge.RECALL, "limit": 3, "include_current_open": True}, [episode_id])])
        self.assertNotIn("episode_id", result)
        self.assertNotIn("episodic", result)

        memory = RealEpisodicMemory(":memory:")
        self.addCleanup(memory.close)
        closed_id = memory.start_episode(bridge.RECALL)
        memory.close_episode(closed_id)
        open_id = memory.start_episode("remember the last time open")
        seen = []
        original = memory.search_episodes

        def wrapped_open(*args, **kwargs):
            found = original(*args, **kwargs)
            seen.append((args, dict(kwargs)))
            return found

        memory.search_episodes = wrapped_open
        with bridge.silence():
            result = self._router(memory).route(bridge.RECALL, {})
        self.assertEqual(seen, [((), {"query": bridge.RECALL, "limit": 3, "include_current_open": True})])
        self.assertEqual(memory.current_episode_id, open_id)
        self.assertNotIn("episode_id", result)
        self.assertNotIn("episodic", result)

        router, recorder = bridge.harness_h()
        router.request_type_proposers = (lambda state: ("learning",),)
        with bridge.silence():
            router.route("ping", {})
        self.assertEqual(
            recorder.calls,
            [("search_episodes", (), {"query": "ping", "limit": 3, "include_current_open": True})],
        )

        created = bridge.make_brain()
        self.addCleanup(created.episodic_memory.close)
        calls = bridge.wrap_episodic(created.episodic_memory)
        with bridge.silence():
            think_result = created.think(bridge.RECALL, use_router=True)
        self.assertEqual(
            bridge.method_names(calls),
            ["start_episode", "search_episodes", "close_episode"],
        )
        self.assertEqual(calls[1][1], ())
        self.assertEqual(calls[1][2], {"query": bridge.RECALL, "limit": 3, "include_current_open": True})
        self.assertNotIn("status", calls[1][2])
        self.assertNotIn("episodes", think_result)

    def test_n9_existing_tests_are_not_edited(self):
        for relative, digest in EXISTING_TEST_SHA256.items():
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(actual, digest, relative)

    def test_n10_like_literal_baseline_file_passes(self):
        catalog = (ROOT / "tools/cognitivo_mutation.py").read_text(encoding="utf-8")
        self.assertIn(LIKE_LITERAL, catalog)
        proc = subprocess.run(
            ["python3", "-m", "unittest", LIKE_LITERAL],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            env={
                "PATH": "/usr/bin:/bin",
                "PYTHONPATH": str(ROOT),
                "NEXUS_AUDIT_DB": "/tmp/nexus_audit_b2b_n10.sqlite3",
                "LC_ALL": "C",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertNotIn("Baseline failed before mutation", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
