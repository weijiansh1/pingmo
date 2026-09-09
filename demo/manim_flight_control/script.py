"""3Blue1Brown-style explainer for the Pingmo flight-control repository."""

from __future__ import annotations

import math

import numpy as np
from manim import (
    BOLD,
    DOWN,
    FadeIn,
    FadeOut,
    GrowArrow,
    GrowFromCenter,
    GrowFromEdge,
    Group,
    Indicate,
    LaggedStart,
    LEFT,
    Line,
    NumberLine,
    NumberPlane,
    ParametricFunction,
    Polygon,
    Rectangle,
    ReplacementTransform,
    RIGHT,
    RoundedRectangle,
    Scene,
    SurroundingRectangle,
    Text,
    UP,
    VGroup,
    WHITE,
    Write,
    Arrow,
    Axes,
    Create,
    DashedLine,
    Dot,
    MoveAlongPath,
    ValueTracker,
    always_redraw,
)


BG = "#111318"
REFERENCE = "#58C4DD"
STUDENT = "#83C167"
TEACHER = "#FFB347"
COMMAND = "#FFFF00"
DANGER = "#FF6B6B"
CONTEXT = "#A7B0BE"
PANEL = "#20242C"
MONO = "Liberation Mono"

TITLE_SIZE = 42
HEADING_SIZE = 30
BODY_SIZE = 25
LABEL_SIZE = 20
CAPTION_SIZE = 22

FAST = 0.7
NORMAL = 1.35
SLOW = 2.2


def label_text(
    value: str,
    *,
    size: int = BODY_SIZE,
    color: str = WHITE,
    weight: str | None = None,
    max_width: float = 12.5,
    t2c: dict[str, str] | None = None,
) -> Text:
    """Create consistently styled text and constrain it to the safe frame."""

    kwargs: dict[str, object] = {
        "font": MONO,
        "font_size": size,
        "color": color,
    }
    if weight is not None:
        kwargs["weight"] = weight
    if t2c is not None:
        kwargs["t2c"] = t2c
    mob = Text(value, **kwargs)
    if mob.width > max_width:
        mob.width = max_width
    return mob


def airplane_icon(color: str, scale: float = 1.0, opacity: float = 1.0) -> Polygon:
    """Return a compact top-view aircraft silhouette pointing right."""

    points = [
        (-1.15, 0.00, 0),
        (-0.72, 0.13, 0),
        (-0.34, 0.14, 0),
        (-0.08, 0.82, 0),
        (0.13, 0.82, 0),
        (0.02, 0.14, 0),
        (0.78, 0.10, 0),
        (1.18, 0.00, 0),
        (0.78, -0.10, 0),
        (0.02, -0.14, 0),
        (0.13, -0.82, 0),
        (-0.08, -0.82, 0),
        (-0.34, -0.14, 0),
        (-0.72, -0.13, 0),
    ]
    return Polygon(
        *points,
        color=color,
        fill_color=color,
        fill_opacity=0.28 * opacity,
        stroke_opacity=opacity,
        stroke_width=2.2,
    ).scale(scale)


def reference_response(t: float, delay: float = 0.22) -> float:
    """Second-order step response used by the repository reference model."""

    if t <= delay:
        return 0.0
    omega_n = 2.0
    zeta = 0.7
    tau = t - delay
    omega_d = omega_n * math.sqrt(1.0 - zeta**2)
    envelope = math.exp(-zeta * omega_n * tau)
    phase = math.cos(omega_d * tau) + zeta / math.sqrt(1.0 - zeta**2) * math.sin(
        omega_d * tau
    )
    return 20.0 * (1.0 - envelope * phase)


def mini_response_chart(
    x_position: float,
    title: str,
    response,
) -> tuple[Group, object]:
    axes = Axes(
        x_range=[0, 5, 1],
        y_range=[-10, 45, 10],
        x_length=3.45,
        y_length=1.75,
        tips=False,
        axis_config={"stroke_color": CONTEXT, "stroke_opacity": 0.18, "stroke_width": 1},
    ).move_to([x_position, -0.65, 0])
    x_label = label_text("t", size=16, color=CONTEXT).next_to(
        axes.x_axis.get_end(), DOWN, buff=0.08
    )
    y_label = label_text("p", size=16, color=CONTEXT).next_to(
        axes.y_axis.get_end(), LEFT, buff=0.08
    )
    case_label = label_text(title, size=19, color=WHITE).next_to(axes, UP, buff=0.18)
    target = DashedLine(
        axes.c2p(0, 20),
        axes.c2p(5, 20),
        color=REFERENCE,
        stroke_opacity=0.42,
        dash_length=0.12,
    )
    graph = axes.plot(response, x_range=[0, 5], color=DANGER, stroke_width=3)
    return Group(axes, x_label, y_label, case_label, target), graph


def metric_row(
    label: str,
    detail: str,
    *,
    color: str,
    width: float = 5.7,
) -> Group:
    frame = RoundedRectangle(
        width=width,
        height=0.78,
        corner_radius=0.12,
        color=color,
        fill_color=PANEL,
        fill_opacity=0.74,
        stroke_width=2,
    )
    left = label_text(label, size=20, color=color, weight=BOLD, max_width=1.5)
    detail_text = label_text(detail, size=17, color=WHITE, max_width=4.2)
    left.move_to(frame.get_left() + RIGHT * 0.72)
    detail_text.move_to(frame.get_center() + RIGHT * 0.55)
    return Group(frame, left, detail_text)


class PingmoScene(Scene):
    """Shared visual and subtitle helpers for all independently renderable scenes."""

    def setup(self) -> None:
        self.camera.background_color = BG

    def title(self, value: str, color: str = WHITE) -> Text:
        return label_text(
            value,
            size=TITLE_SIZE,
            color=color,
            weight=BOLD,
            max_width=12.7,
        ).to_edge(UP, buff=0.55)

    def caption(self, value: str, color: str = WHITE) -> Text:
        mob = label_text(value, size=CAPTION_SIZE, color=color, max_width=12.3)
        mob.set_stroke(BG, width=6, background=True)
        return mob.to_edge(DOWN, buff=0.55)

    def narrated_play(
        self,
        narration: str,
        *animations,
        run_time: float = NORMAL,
        hold: float = 1.0,
    ) -> None:
        self.add_subcaption(narration, duration=run_time + hold)
        self.play(*animations, run_time=run_time)
        self.wait(hold)

    def replace_caption(
        self,
        old: Text,
        value: str,
        *,
        color: str = WHITE,
        run_time: float = FAST,
        hold: float = 1.0,
    ) -> Text:
        new = self.caption(value, color=color)
        self.narrated_play(
            value,
            ReplacementTransform(old, new),
            run_time=run_time,
            hold=hold,
        )
        return new

    def clean_exit(self) -> None:
        if self.mobjects:
            self.play(FadeOut(Group(*self.mobjects)), run_time=0.5)
        self.wait(0.3)


class Scene1_SameCommand(PingmoScene):
    def construct(self) -> None:
        title = self.title("同一个滚转指令，为什么三架飞机给出三种答案？", COMMAND)
        self.narrated_play(
            "同一个滚转角速度指令，为什么三架飞机会给出三种答案？",
            Write(title),
            run_time=1.5,
            hold=1.4,
        )

        command_label = label_text(
            "同一个指令   p_c = 20 deg/s", size=27, color=COMMAND, weight=BOLD
        ).move_to([0, 1.72, 0])
        command_stem = Arrow(
            [0, 1.38, 0], [0, 0.92, 0], color=COMMAND, buff=0.02, stroke_width=3
        )
        plane_positions = [np.array([-4.35, 0.32, 0]), np.array([0, 0.32, 0]), np.array([4.35, 0.32, 0])]
        planes = Group(
            *[
                airplane_icon(color, scale=0.44).move_to(position)
                for color, position in zip(
                    [REFERENCE, TEACHER, STUDENT], plane_positions, strict=True
                )
            ]
        )
        branches = VGroup(
            *[
                Arrow(
                    [0, 0.9, 0],
                    position + UP * 0.44,
                    color=COMMAND,
                    stroke_opacity=0.62,
                    stroke_width=2,
                    buff=0.05,
                )
                for position in plane_positions
            ]
        )
        caption = self.caption("同一条指令，被送进三个不同的动力学对象。")
        self.narrated_play(
            "同一条指令，被送进三个不同的动力学对象。",
            FadeIn(command_label),
            GrowArrow(command_stem),
            FadeIn(caption),
            run_time=1.0,
            hold=0.6,
        )
        self.narrated_play(
            "对象的固有频率、阻尼和纯时延都不相同。",
            LaggedStart(*[GrowArrow(branch) for branch in branches], lag_ratio=0.16),
            LaggedStart(*[GrowFromCenter(plane) for plane in planes], lag_ratio=0.2),
            run_time=1.7,
            hold=1.1,
        )

        raw_responses = [
            lambda t: 20.0 * (1.0 - math.exp(-0.33 * t)),
            lambda t: 20.0
            * (1.0 - math.exp(-0.36 * t) * (math.cos(2.6 * t) + 0.12 * math.sin(2.6 * t))),
            lambda t: 20.0 * (1.0 - math.exp(-0.12 * t) * math.cos(4.0 * t)),
        ]
        chart_specs = [
            mini_response_chart(-4.35, "对象 A：慢", raw_responses[0]),
            mini_response_chart(0, "对象 B：振荡", raw_responses[1]),
            mini_response_chart(4.35, "对象 C：强超调", raw_responses[2]),
        ]
        chart_bases = [item[0] for item in chart_specs]
        graphs = [item[1] for item in chart_specs]
        self.narrated_play(
            "不加适配控制时，响应速度、振荡和峰值完全不同。",
            FadeOut(Group(command_label, command_stem, branches, planes)),
            LaggedStart(*[FadeIn(base) for base in chart_bases], lag_ratio=0.16),
            run_time=1.25,
            hold=0.5,
        )
        self.narrated_play(
            "蓝色虚线是同一个目标，红色曲线却各走各的。",
            LaggedStart(*[Create(graph) for graph in graphs], lag_ratio=0.22),
            run_time=2.3,
            hold=1.3,
        )
        caption = self.replace_caption(
            caption,
            "控制器面对的不是一架飞机，而是一族 G(theta)。",
            color=COMMAND,
            hold=1.5,
        )
        self.narrated_play(
            "控制问题的难点，正来自这一个对象族。",
            LaggedStart(*[Indicate(graph, color=DANGER) for graph in graphs], lag_ratio=0.2),
            run_time=1.4,
            hold=1.3,
        )
        self.clean_exit()


class Scene2_SameQuality(PingmoScene):
    def construct(self) -> None:
        title = self.title("统一响应，不是统一动作", REFERENCE)
        self.narrated_play(
            "先定义什么样的响应，才叫飞得像目标飞机。",
            FadeIn(title, shift=UP * 0.2),
            run_time=1.2,
            hold=1.0,
        )

        p_command = label_text("p_c", size=27, color=COMMAND, weight=BOLD)
        model_box = RoundedRectangle(
            width=2.8,
            height=0.82,
            corner_radius=0.12,
            color=REFERENCE,
            fill_color=PANEL,
            fill_opacity=0.86,
        )
        model_text = label_text("二阶参考模型", size=22, color=REFERENCE).move_to(model_box)
        p_reference = label_text("p_ref", size=27, color=REFERENCE, weight=BOLD)
        model_group = Group(model_box, model_text)
        pipeline = Group(p_command, model_group, p_reference).arrange(RIGHT, buff=0.75)
        pipeline.move_to([-1.0, 2.02, 0])
        arrow_1 = Arrow(
            p_command.get_right(), model_box.get_left(), color=COMMAND, buff=0.12
        )
        arrow_2 = Arrow(
            model_box.get_right(), p_reference.get_left(), color=REFERENCE, buff=0.12
        )
        model_params = label_text(
            "omega_n = 2.0 rad/s    zeta = 0.7",
            size=17,
            color=CONTEXT,
            max_width=4.0,
        ).next_to(model_box, DOWN, buff=0.16)
        caption = self.caption("先用一个二阶模型，画出唯一的目标响应。")
        self.narrated_play(
            "先用一个二阶模型，画出唯一的目标响应。",
            FadeIn(p_command),
            FadeIn(caption),
            run_time=0.8,
            hold=0.4,
        )
        self.narrated_play(
            "参考模型的自然频率是二点零，阻尼比是零点七。",
            GrowArrow(arrow_1),
            GrowFromCenter(model_group),
            FadeIn(model_params),
            run_time=1.35,
            hold=0.8,
        )
        self.narrated_play(
            "它产生参考滚转角速度 p_ref。",
            GrowArrow(arrow_2),
            FadeIn(p_reference),
            run_time=1.0,
            hold=0.8,
        )

        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 26, 5],
            x_length=8.0,
            y_length=3.05,
            tips=False,
            axis_config={"stroke_color": CONTEXT, "stroke_opacity": 0.17, "stroke_width": 1},
        ).move_to([-1.45, -0.38, 0])
        x_label = label_text("t", size=17, color=CONTEXT).next_to(
            axes.x_axis.get_end(), DOWN, buff=0.1
        )
        y_label = label_text("p (deg/s)", size=16, color=CONTEXT).rotate(math.pi / 2)
        y_label.next_to(axes.y_axis, LEFT, buff=0.28)
        target = axes.plot(reference_response, x_range=[0, 5], color=REFERENCE, stroke_width=4)
        target_label = label_text("目标 p_ref", size=18, color=REFERENCE).move_to(
            axes.c2p(4.15, 22.7)
        )
        self.narrated_play(
            "目标先以几何形状出现：快速、平稳、几乎没有超调。",
            Create(axes),
            FadeIn(Group(x_label, y_label)),
            Create(target),
            FadeIn(target_label),
            run_time=2.1,
            hold=1.0,
        )

        action_title = label_text("不同的 F_as(t)", size=19, color=STUDENT).move_to(
            [5.05, 1.25, 0]
        )
        action_rows = Group()
        for index, (name, length) in enumerate(zip(["A", "B", "C"], [0.85, 1.45, 2.05], strict=True)):
            y_position = 0.62 - index * 0.72
            name_label = label_text(name, size=18, color=WHITE).move_to([4.0, y_position, 0])
            force_arrow = Arrow(
                [4.3, y_position, 0],
                [4.3 + length, y_position, 0],
                color=STUDENT,
                buff=0,
                stroke_width=3,
                max_tip_length_to_length_ratio=0.18,
            )
            action_rows.add(Group(name_label, force_arrow))
        self.narrated_play(
            "三架飞机需要三条不同的控制动作。",
            FadeIn(action_title),
            LaggedStart(
                *[
                    FadeIn(row[0]) if animation_index % 2 == 0 else GrowArrow(row[1])
                    for row in action_rows
                    for animation_index in range(2)
                ],
                lag_ratio=0.12,
            ),
            run_time=1.55,
            hold=0.8,
        )
        caption = self.replace_caption(
            caption,
            "动作可以不同；三条闭环响应必须靠近同一个目标。",
            color=STUDENT,
            hold=0.5,
        )

        controlled_functions = [
            lambda t: reference_response(t, delay=0.28) * (1.0 - 0.025 * math.exp(-0.6 * t)),
            lambda t: reference_response(t, delay=0.20) + 0.55 * math.exp(-0.75 * t) * math.sin(3.0 * t),
            lambda t: reference_response(t, delay=0.16) - 0.7 * math.exp(-0.82 * t) * math.sin(2.4 * t),
        ]
        controlled = [
            axes.plot(
                function,
                x_range=[0, 5],
                color=STUDENT,
                stroke_width=2.4,
                stroke_opacity=opacity,
            )
            for function, opacity in zip(controlled_functions, [0.48, 0.72, 1.0], strict=True)
        ]
        self.narrated_play(
            "动作不同，但三条闭环响应都被拉向同一个目标。",
            LaggedStart(*[Create(graph) for graph in controlled], lag_ratio=0.18),
            run_time=2.25,
            hold=1.2,
        )
        equation = label_text(
            "pi(o, theta)  ->  F_as          p  ->  p_ref",
            size=25,
            color=WHITE,
            weight=BOLD,
            max_width=9.0,
            t2c={"theta": COMMAND, "F_as": STUDENT, "p_ref": REFERENCE},
        ).move_to([-1.0, -2.32, 0])
        self.narrated_play(
            "代数只是这幅图的压缩写法。",
            FadeIn(equation, shift=UP * 0.15),
            run_time=1.1,
            hold=1.0,
        )
        caption = self.replace_caption(
            caption,
            "通用控制的目标：动作可以不同，飞行品质要一致。",
            color=COMMAND,
            hold=1.4,
        )
        self.clean_exit()


class Scene3_TeacherBank(PingmoScene):
    def construct(self) -> None:
        title = self.title("让 32 个专家，教会一个通用控制器", TEACHER)
        self.narrated_play(
            "仓库先为每架固定飞机，训练一个纯奖励 TD3 Teacher。",
            Write(title),
            run_time=1.35,
            hold=1.0,
        )

        plane = NumberPlane(
            x_range=[0, 8, 1],
            y_range=[0, 4, 1],
            x_length=5.2,
            y_length=3.65,
            background_line_style={
                "stroke_color": CONTEXT,
                "stroke_opacity": 0.12,
                "stroke_width": 1,
            },
            axis_config={"stroke_opacity": 0.2, "stroke_color": CONTEXT},
        ).move_to([-3.58, -0.12, 0])
        theta_x = label_text("theta_a", size=17, color=CONTEXT).next_to(
            plane.x_axis.get_end(), DOWN, buff=0.08
        )
        theta_y = label_text("theta_b", size=17, color=CONTEXT).move_to(
            plane.get_corner(UP + LEFT) + RIGHT * 0.48 + DOWN * 0.2
        )
        plane_label = label_text(
            "飞机参数空间", size=22, color=CONTEXT, weight=BOLD
        ).next_to(plane, UP, buff=0.22)
        dots = VGroup()
        for index in range(32):
            x_value = index % 8 + 0.42 + 0.12 * math.sin(index * 1.7)
            y_value = index // 8 + 0.42 + 0.12 * math.cos(index * 1.3)
            dots.add(
                Dot(
                    plane.c2p(x_value, y_value),
                    radius=0.07,
                    color=TEACHER,
                    fill_opacity=0.95,
                )
            )
        caption = self.caption("每个橙色点，都是一架飞机和它自己的 Teacher。")
        self.narrated_play(
            "每个橙色点，都是一架飞机和它自己的 Teacher。",
            Create(plane),
            FadeIn(Group(theta_x, theta_y, plane_label, caption)),
            run_time=1.2,
            hold=0.6,
        )
        self.narrated_play(
            "六十八次尝试中，三十二个 Teacher 通过质量门禁。",
            LaggedStart(*[GrowFromCenter(dot) for dot in dots], lag_ratio=0.035),
            run_time=2.0,
            hold=1.2,
        )
        bank_count = label_text(
            "32 个合格 Teacher", size=24, color=TEACHER, weight=BOLD
        ).move_to([-3.58, -2.35, 0])
        self.narrated_play(
            "每个 Teacher 只需要精通自己的固定参数 theta_i。",
            FadeIn(bank_count, shift=UP * 0.15),
            run_time=0.9,
            hold=0.8,
        )

        student_frame = RoundedRectangle(
            width=4.15,
            height=3.35,
            corner_radius=0.16,
            color=STUDENT,
            fill_color=PANEL,
            fill_opacity=0.9,
            stroke_width=2.5,
        ).move_to([3.75, -0.05, 0])
        student_title = label_text(
            "Dense Student", size=28, color=STUDENT, weight=BOLD
        ).move_to(student_frame.get_top() + DOWN * 0.48)
        input_text = label_text(
            "输入   35-d o  +  8-d theta", size=19, color=WHITE, max_width=3.65
        ).move_to(student_frame.get_center() + UP * 0.38)
        output_text = label_text(
            "输出   requested F_as", size=19, color=WHITE, max_width=3.65
        ).move_to(student_frame.get_center() + DOWN * 0.35)
        parameter_text = label_text(
            "540,417 参数", size=21, color=COMMAND, weight=BOLD
        ).move_to(student_frame.get_bottom() + UP * 0.48)
        student_shell = Group(student_frame, student_title)
        distill_arrow = Arrow(
            plane.get_right() + RIGHT * 0.08,
            student_frame.get_left() + LEFT * 0.08,
            color=TEACHER,
            stroke_width=4,
            buff=0.08,
        )
        distill_label = label_text(
            "动作标签", size=18, color=TEACHER
        ).next_to(distill_arrow, UP, buff=0.15)
        compression = label_text(
            "32  ->  1", size=31, color=COMMAND, weight=BOLD
        ).move_to([0.4, 2.0, 0])
        self.narrated_play(
            "现在把三十二套专家动作，汇入一个条件 Student。",
            GrowArrow(distill_arrow),
            FadeIn(distill_label),
            GrowFromCenter(student_shell),
            run_time=1.45,
            hold=0.9,
        )
        self.narrated_play(
            "Student 同时读取三十五维控制状态和八维飞机参数。",
            FadeIn(input_text, shift=RIGHT * 0.18),
            run_time=1.0,
            hold=0.8,
        )
        self.narrated_play(
            "它直接输出完整的 requested F_as，而不是 PID 残差。",
            FadeIn(output_text, shift=RIGHT * 0.18),
            run_time=1.0,
            hold=0.8,
        )
        self.narrated_play(
            "最终得到一个五十四万参数的通用策略。",
            FadeIn(parameter_text),
            GrowFromCenter(compression),
            run_time=1.3,
            hold=1.2,
        )
        caption = self.replace_caption(
            caption,
            "Teacher 精通一架飞机；Student 学会根据 theta 改变控制规律。",
            color=COMMAND,
            hold=1.8,
        )
        self.clean_exit()


class Scene4_DAgger(PingmoScene):
    def construct(self) -> None:
        title = self.title("让 Student 去它真正会犯错的地方", STUDENT)
        self.narrated_play(
            "只模仿 Teacher 的完美轨迹，闭环里仍然会遇到分布漂移。",
            FadeIn(title, shift=LEFT * 0.2),
            run_time=1.2,
            hold=1.0,
        )

        axes = Axes(
            x_range=[-2.5, 2.5, 1],
            y_range=[-1.5, 2.5, 1],
            x_length=6.0,
            y_length=4.25,
            tips=False,
            axis_config={"stroke_color": CONTEXT, "stroke_opacity": 0.18, "stroke_width": 1},
        ).move_to([-3.45, -0.15, 0])
        x_label = label_text("状态误差 e", size=17, color=CONTEXT).next_to(
            axes.x_axis.get_end(), DOWN, buff=0.12
        )
        y_label = label_text("e_dot", size=17, color=CONTEXT).next_to(
            axes.y_axis.get_end(), LEFT, buff=0.12
        )

        def teacher_point(t: float) -> np.ndarray:
            return axes.c2p(-2.15 + 4.3 * t, 0.36 * math.sin(math.pi * t))

        def drifting_point(t: float) -> np.ndarray:
            return axes.c2p(
                -2.15 + 4.3 * t,
                0.32 * math.sin(math.pi * t) + 1.75 * t**2,
            )

        def corrected_point(t: float) -> np.ndarray:
            return axes.c2p(
                -2.15 + 4.3 * t,
                0.34 * math.sin(math.pi * t) + 0.12 * t,
            )

        teacher_path = ParametricFunction(
            teacher_point, t_range=[0, 1], color=TEACHER, stroke_width=4
        )
        student_path = ParametricFunction(
            drifting_point, t_range=[0, 1], color=DANGER, stroke_width=4
        )
        corrected_path = ParametricFunction(
            corrected_point, t_range=[0, 1], color=STUDENT, stroke_width=4
        )
        teacher_samples = VGroup(
            *[
                Dot(teacher_point(t), radius=0.055, color=TEACHER)
                for t in np.linspace(0.06, 0.94, 13)
            ]
        )
        state_labels = VGroup(
            *[
                Dot(
                    drifting_point(t),
                    radius=0.105,
                    color=TEACHER,
                    fill_opacity=0.18,
                    stroke_width=2,
                )
                for t in [0.38, 0.52, 0.66, 0.80, 0.94]
            ]
        )
        teacher_tag = label_text(
            "Teacher 驱动数据", size=19, color=TEACHER
        ).move_to([-3.65, 2.35, 0])
        caption = self.caption("第一轮只在 Teacher 访问到的状态上学习。")
        self.narrated_play(
            "第一轮只在 Teacher 访问到的状态上学习。",
            Create(axes),
            FadeIn(Group(x_label, y_label, teacher_tag, caption)),
            run_time=1.15,
            hold=0.5,
        )
        self.narrated_play(
            "这些橙色样本覆盖的是专家的窄状态走廊。",
            Create(teacher_path),
            LaggedStart(*[GrowFromCenter(dot) for dot in teacher_samples], lag_ratio=0.06),
            run_time=1.8,
            hold=0.9,
        )
        student_dot = Dot(drifting_point(0), radius=0.09, color=STUDENT)
        self.add(student_dot)
        self.narrated_play(
            "Student 一旦产生小误差，就会走进训练集外的状态。",
            MoveAlongPath(student_dot, student_path),
            Create(student_path),
            run_time=2.25,
            hold=1.1,
        )

        dataset_header = label_text(
            "累计样本", size=21, color=CONTEXT, weight=BOLD
        ).move_to([3.55, 1.82, 0])
        round_zero = label_text(
            "264k", size=37, color=TEACHER, weight=BOLD
        ).move_to([3.55, 1.08, 0])
        driver_label = label_text(
            "Teacher-driven", size=18, color=TEACHER
        ).next_to(round_zero, DOWN, buff=0.16)
        self.narrated_play(
            "Teacher 驱动的初始数据集有二十六万四千行。",
            FadeIn(Group(dataset_header, round_zero, driver_label)),
            run_time=0.9,
            hold=0.7,
        )
        self.narrated_play(
            "DAgger 改让 Student 驱动，再请对应 Teacher 标注这些新状态。",
            LaggedStart(*[GrowFromCenter(dot) for dot in state_labels], lag_ratio=0.1),
            run_time=1.4,
            hold=0.8,
        )
        caption = self.replace_caption(
            caption,
            "Student 驱动闭环状态，再由对应 Teacher 重新标注。",
            color=STUDENT,
            hold=0.5,
        )
        round_one = label_text(
            "528k", size=37, color=STUDENT, weight=BOLD
        ).move_to(round_zero)
        student_driver = label_text(
            "Student-driven + Teacher 标签", size=17, color=STUDENT, max_width=4.2
        ).move_to(driver_label)
        self.narrated_play(
            "第一轮 Student-driven 聚合后，样本翻倍到五十二万八千行。",
            ReplacementTransform(round_zero, round_one),
            ReplacementTransform(driver_label, student_driver),
            ReplacementTransform(student_path, corrected_path),
            student_dot.animate.move_to(corrected_point(1)).set_color(STUDENT),
            run_time=1.7,
            hold=1.0,
        )
        round_two = label_text(
            "792k", size=37, color=CONTEXT, weight=BOLD
        ).move_to(round_one)
        self.narrated_play(
            "再聚合一轮，数据增加到七十九万两千行。",
            ReplacementTransform(round_one, round_two),
            run_time=0.9,
            hold=0.7,
        )

        self.play(
            FadeOut(Group(dataset_header, round_two, student_driver)),
            run_time=0.5,
        )
        round_1 = metric_row(
            "R1", "闭环 RMSE 1.079   |   TV 212.8", color=STUDENT
        ).move_to([3.55, 0.72, 0])
        round_2 = metric_row(
            "R2", "闭环 RMSE 1.091   |   TV 367.6", color=DANGER
        ).move_to([3.55, -0.48, 0])
        offline_badge = label_text(
            "R2 离线 gap 更小：0.053", size=17, color=CONTEXT
        ).next_to(round_2, DOWN, buff=0.22)
        selected = label_text(
            "SELECTED", size=18, color=STUDENT, weight=BOLD
        ).next_to(round_1, UP, buff=0.16)
        self.narrated_play(
            "但是最终选择的不是离线拟合最接近 Teacher 的第二轮。",
            FadeIn(round_1, shift=LEFT * 0.2),
            FadeIn(round_2, shift=LEFT * 0.2),
            FadeIn(offline_badge),
            run_time=1.25,
            hold=0.9,
        )
        self.narrated_play(
            "第一轮的闭环 RMSE 更低，动作总变差也小得多。",
            FadeIn(selected),
            Indicate(round_1, color=STUDENT),
            run_time=1.2,
            hold=1.0,
        )
        caption = self.replace_caption(
            caption,
            "闭环最好，不等于离线 action loss 最小。",
            color=COMMAND,
            hold=1.8,
        )
        self.clean_exit()


class Scene5_UnseenResults(PingmoScene):
    def _axis_group(
        self,
        *,
        x_left: float,
        x_right: float,
        baseline: float,
        height: float,
        max_value: float,
        tick_values: list[float],
    ) -> Group:
        y_axis = Line(
            [x_left, baseline, 0],
            [x_left, baseline + height, 0],
            color=CONTEXT,
            stroke_opacity=0.4,
        )
        x_axis = Line(
            [x_left, baseline, 0],
            [x_right, baseline, 0],
            color=CONTEXT,
            stroke_opacity=0.4,
        )
        ticks = VGroup()
        numbers = Group()
        for value in tick_values:
            y = baseline + height * value / max_value
            ticks.add(
                Line(
                    [x_left - 0.08, y, 0],
                    [x_right, y, 0],
                    color=CONTEXT,
                    stroke_opacity=0.11,
                    stroke_width=1,
                )
            )
            numbers.add(
                label_text(f"{value:g}", size=15, color=CONTEXT).move_to(
                    [x_left - 0.34, y, 0]
                )
            )
        y_label = label_text("mean RMSE", size=15, color=CONTEXT).rotate(math.pi / 2)
        y_label.move_to([x_left - 0.82, baseline + height / 2, 0])
        return Group(y_axis, x_axis, ticks, numbers, y_label)

    @staticmethod
    def _bar(
        x: float,
        baseline: float,
        value: float,
        max_value: float,
        height: float,
        color: str,
        name: str,
    ) -> tuple[Rectangle, Group]:
        bar_height = max(0.06, height * value / max_value)
        bar = Rectangle(
            width=0.82,
            height=bar_height,
            stroke_width=0,
            fill_color=color,
            fill_opacity=0.9,
        ).move_to([x, baseline + bar_height / 2, 0])
        value_label = label_text(
            f"{value:.2f}", size=17, color=color, weight=BOLD
        ).next_to(bar, UP, buff=0.12)
        name_label = label_text(name, size=16, color=WHITE).move_to(
            [x, baseline - 0.28, 0]
        )
        return bar, Group(value_label, name_label)

    def construct(self) -> None:
        title = self.title("完全未见飞机：冻结 Student，零适配", STUDENT)
        self.narrated_play(
            "最终把 Student 冻结，在十架完全未见飞机上测试。",
            Write(title),
            run_time=1.35,
            hold=1.0,
        )
        scope = label_text(
            "10 架未见飞机   |   60 个飞机-命令对   |   30 s 测试窗口",
            size=20,
            color=CONTEXT,
            max_width=10.5,
        ).move_to([0, 2.35, 0])
        caption = self.caption("先看原始对象与 v3 Student 的量级差异。")
        self.narrated_play(
            "测试集合与训练飞机没有重叠，Student checkpoint 也保持不变。",
            FadeIn(scope),
            FadeIn(caption),
            run_time=0.9,
            hold=0.7,
        )

        baseline = -1.35
        chart_height = 3.35
        full_axis = self._axis_group(
            x_left=-5.75,
            x_right=-1.25,
            baseline=baseline,
            height=chart_height,
            max_value=50.0,
            tick_values=[0, 25, 50],
        )
        raw_bar, raw_labels = self._bar(
            -4.65, baseline, 48.377, 50.0, chart_height, DANGER, "Raw"
        )
        v3_full_bar, v3_full_labels = self._bar(
            -2.65, baseline, 2.411, 50.0, chart_height, STUDENT, "v3"
        )
        self.narrated_play(
            "原始对象的平均跟踪 RMSE 是四十八点三八度每秒。",
            FadeIn(full_axis),
            GrowFromEdge(raw_bar, DOWN),
            FadeIn(raw_labels),
            run_time=1.55,
            hold=0.8,
        )
        improvement = label_text(
            "~20x lower", size=20, color=COMMAND, weight=BOLD
        ).move_to([-3.62, 1.52, 0])
        self.narrated_play(
            "v3 Student 将平均 RMSE 降到二点四一，约低二十倍。",
            GrowFromEdge(v3_full_bar, DOWN),
            FadeIn(v3_full_labels),
            FadeIn(improvement),
            run_time=1.45,
            hold=1.0,
        )

        zoom_axis = self._axis_group(
            x_left=0.2,
            x_right=6.1,
            baseline=baseline,
            height=chart_height,
            max_value=5.0,
            tick_values=[0, 2.5, 5.0],
        )
        v2_bar, v2_labels = self._bar(
            1.35, baseline, 4.442, 5.0, chart_height, CONTEXT, "v2"
        )
        v3_bar, v3_labels = self._bar(
            3.15, baseline, 2.411, 5.0, chart_height, STUDENT, "v3"
        )
        pid_bar, pid_labels = self._bar(
            4.95, baseline, 1.409, 5.0, chart_height, REFERENCE, "逐机 PID"
        )
        self.narrated_play(
            "放大后可以看到，v3 比 v2 明显进步。",
            FadeIn(zoom_axis),
            LaggedStart(
                GrowFromEdge(v2_bar, DOWN),
                GrowFromEdge(v3_bar, DOWN),
                lag_ratio=0.24,
            ),
            FadeIn(Group(v2_labels, v3_labels)),
            run_time=1.65,
            hold=0.9,
        )
        self.narrated_play(
            "但它仍落后于每架飞机单独整定的 PID。",
            GrowFromEdge(pid_bar, DOWN),
            FadeIn(pid_labels),
            run_time=1.05,
            hold=1.0,
        )

        raw_tracker = ValueTracker(0)
        v2_tracker = ValueTracker(0)
        counter_raw = always_redraw(
            lambda: label_text(
                f"{raw_tracker.get_value():.2f}%",
                size=26,
                color=STUDENT,
                weight=BOLD,
            ).move_to([-2.55, -2.25, 0])
        )
        counter_v2 = always_redraw(
            lambda: label_text(
                f"{v2_tracker.get_value():.0f}%",
                size=26,
                color=COMMAND,
                weight=BOLD,
            ).move_to([2.55, -2.25, 0])
        )
        raw_note = label_text("58/60 优于 Raw", size=16, color=STUDENT).next_to(
            counter_raw, RIGHT, buff=0.34
        )
        v2_note = label_text("51/60 优于 v2", size=16, color=COMMAND).next_to(
            counter_v2, RIGHT, buff=0.34
        )
        self.add(counter_raw, counter_v2)
        self.narrated_play(
            "六十个测试对里，百分之九十六点六七优于 Raw，百分之八十五优于 v2。",
            raw_tracker.animate.set_value(96.67),
            v2_tracker.animate.set_value(85),
            FadeIn(raw_note),
            FadeIn(v2_note),
            run_time=2.2,
            hold=1.3,
        )
        caption = self.replace_caption(
            caption,
            "一个冻结的 Student，已在绝大多数未见测试对上改善原始响应。",
            color=STUDENT,
            hold=1.7,
        )
        self.clean_exit()


class Scene6_HonestBoundary(PingmoScene):
    def _gauge(
        self,
        *,
        y: float,
        maximum: float,
        step: float,
        threshold: float,
        observed: float,
        heading: str,
        observed_label: str,
        threshold_label: str,
        include_numbers: bool,
    ) -> tuple[Group, Dot, Text]:
        line = NumberLine(
            x_range=[0, maximum, step],
            length=8.0,
            include_numbers=False,
            font_size=16,
            color=CONTEXT,
            stroke_opacity=0.42,
            include_tip=False,
        ).move_to([0, y, 0])
        heading_text = label_text(
            heading, size=20, color=WHITE, weight=BOLD
        ).next_to(line, UP, buff=0.33)
        threshold_point = line.n2p(threshold)
        threshold_mark = Line(
            threshold_point + DOWN * 0.2,
            threshold_point + UP * 0.2,
            color=COMMAND,
            stroke_width=4,
        )
        threshold_text = label_text(
            threshold_label, size=16, color=COMMAND
        ).next_to(threshold_mark, DOWN, buff=0.16)
        dot = Dot(line.n2p(0), radius=0.11, color=DANGER)
        observed_text = label_text(
            observed_label, size=18, color=DANGER, weight=BOLD
        ).move_to(line.n2p(observed) + UP * 0.33)
        number_labels = Group()
        if include_numbers:
            value = 0.0
            while value <= maximum + 1e-9:
                number_labels.add(
                    label_text(f"{value:g}", size=15, color=CONTEXT).next_to(
                        line.n2p(value), DOWN, buff=0.14
                    )
                )
                value += step
        group = Group(
            line,
            heading_text,
            threshold_mark,
            threshold_text,
            number_labels,
        )
        return group, dot, observed_text

    def construct(self) -> None:
        title = self.title("有效，不等于合格", DANGER)
        self.narrated_play(
            "这条路线已经有效，但正式质量门禁仍然失败。",
            Write(title),
            run_time=1.4,
            hold=1.1,
        )
        summary = label_text(
            "一个控制器，开始覆盖一类飞机", size=27, color=STUDENT, weight=BOLD
        ).move_to([0, 2.18, 0])
        caption = self.caption("最后看两道没有通过的工程门槛。")
        self.narrated_play(
            "一个条件 Student 已经开始覆盖一类飞机。",
            FadeIn(summary, shift=UP * 0.18),
            FadeIn(caption),
            run_time=1.0,
            hold=0.8,
        )

        peak_group, peak_dot, peak_text = self._gauge(
            y=0.65,
            maximum=25,
            step=5,
            threshold=5,
            observed=22.906,
            heading="最大峰值误差 (deg/s)",
            observed_label="22.906",
            threshold_label="门禁 <= 5",
            include_numbers=True,
        )
        self.narrated_play(
            "最大峰值误差是二十二点九一，门禁只允许五。",
            FadeIn(peak_group),
            GrowFromCenter(peak_dot),
            run_time=1.0,
            hold=0.35,
        )
        peak_line = peak_group[0]
        self.narrated_play(
            "观测值远远越过阈值。",
            peak_dot.animate.move_to(peak_line.n2p(22.906)),
            FadeIn(peak_text),
            run_time=1.55,
            hold=0.8,
        )

        tv_group, tv_dot, tv_text = self._gauge(
            y=-1.0,
            maximum=2.0,
            step=0.25,
            threshold=1.25,
            observed=1.816,
            heading="Student / Teacher requested-force TV",
            observed_label="1.816",
            threshold_label="门禁 <= 1.25",
            include_numbers=False,
        )
        tv_line = tv_group[0]
        scale_labels = Group(
            label_text("0", size=15, color=CONTEXT).next_to(
                tv_line.n2p(0), DOWN, buff=0.14
            ),
            label_text("2.0", size=15, color=CONTEXT).next_to(
                tv_line.n2p(2.0), DOWN, buff=0.14
            ),
        )
        self.narrated_play(
            "动作总变差比 Teacher 高一点八一六倍，也超过一点二五的门禁。",
            FadeIn(tv_group),
            FadeIn(scale_labels),
            GrowFromCenter(tv_dot),
            run_time=1.0,
            hold=0.35,
        )
        self.narrated_play(
            "这意味着 Student 仍有明显的动作抖动。",
            tv_dot.animate.move_to(tv_line.n2p(1.816)),
            FadeIn(tv_text),
            run_time=1.4,
            hold=0.8,
        )
        failed = label_text(
            "quality_gate_failed", size=24, color=DANGER, weight=BOLD
        ).move_to([0, -2.35, 0])
        failed_frame = SurroundingRectangle(
            failed, color=DANGER, buff=0.2, corner_radius=0.08
        )
        self.narrated_play(
            "所以仓库把正式状态写成 quality gate failed，而不是 complete。",
            FadeIn(Group(failed_frame, failed)),
            run_time=1.1,
            hold=1.0,
        )

        caption = self.replace_caption(
            caption,
            "它证明了路线有效，也精确指出了下一步该修什么。",
            color=COMMAND,
            hold=0.8,
        )
        self.narrated_play(
            "下一步不是继续包装指标，而是处理高峰值状态和动作平滑性。",
            FadeOut(
                Group(
                    title,
                    summary,
                    peak_group,
                    peak_dot,
                    peak_text,
                    tv_group,
                    tv_dot,
                    tv_text,
                    scale_labels,
                    failed_frame,
                    failed,
                )
            ),
            run_time=0.75,
            hold=0.4,
        )
        project = label_text(
            "PINGMO  /  FLIGHT RL CONTROL", size=20, color=CONTEXT, weight=BOLD
        ).move_to([0, 1.65, 0])
        takeaway = label_text(
            "一个控制器，开始控制一类飞机", size=38, color=STUDENT, weight=BOLD
        ).move_to([0, 0.45, 0])
        honest = label_text(
            "有效  !=  合格", size=28, color=COMMAND, weight=BOLD
        ).move_to([0, -0.55, 0])
        next_step = label_text(
            "下一步：高峰值状态  +  动作平滑性", size=22, color=WHITE
        ).move_to([0, -1.45, 0])
        self.narrated_play(
            "一个控制器控制一类飞机，这个目标已经从问题变成了可验证的工程路线。",
            FadeIn(project),
            GrowFromCenter(takeaway),
            FadeIn(honest),
            FadeIn(next_step),
            run_time=1.8,
            hold=2.8,
        )
        self.clean_exit()
