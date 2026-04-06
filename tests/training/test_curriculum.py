from latent_consensus.training.curriculum import CurriculumSchedule, CurriculumStage


def test_curriculum_schedule_returns_expected_stage() -> None:
    schedule = CurriculumSchedule(
        stages=[
            CurriculumStage(start_epoch=0, label="2-step"),
            CurriculumStage(start_epoch=3, label="4-step"),
            CurriculumStage(start_epoch=5, label="6-step"),
        ]
    )

    assert schedule.stage_for_epoch(0).label == "2-step"
    assert schedule.stage_for_epoch(4).label == "4-step"
    assert schedule.stage_for_epoch(7).label == "6-step"


def test_curriculum_schedule_requires_non_empty_stages() -> None:
    try:
        CurriculumSchedule(stages=[])
    except ValueError as error:
        assert "至少需要一个阶段" in str(error)
    else:
        raise AssertionError("空 curriculum 不应通过")
