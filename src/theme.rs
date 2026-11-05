//! OVERWORLD-inspired look: warm parchment, ink outlines, SNES-style hard shadows,
//! Pixelify Sans display type, Inter body, VT323 readouts.

use egui::{
    Color32, CornerRadius, FontData, FontDefinitions, FontFamily, FontId, Frame, Margin, Response,
    RichText, Sense, Shadow, Stroke, StrokeKind, TextStyle, Ui, Vec2, vec2,
};
use std::sync::Arc;

pub const PARCHMENT: Color32 = Color32::from_rgb(0xF2, 0xEA, 0xD6);
pub const BONE: Color32 = Color32::from_rgb(0xFF, 0xFA, 0xEF);
pub const INK: Color32 = Color32::from_rgb(0x16, 0x13, 0x10);
pub const COBALT: Color32 = Color32::from_rgb(0x2E, 0x5D, 0xD6);
pub const VERMILLION: Color32 = Color32::from_rgb(0xE2, 0x52, 0x2E);
pub const PINE: Color32 = Color32::from_rgb(0x2F, 0x6E, 0x4F);
pub const TAN: Color32 = Color32::from_rgb(0xD9, 0xCD, 0xB3);
pub const SKY: Color32 = Color32::from_rgb(0x5A, 0x9A, 0xB7);
/// Ink at reduced strength for secondary text on parchment.
pub const INK_SOFT: Color32 = Color32::from_rgb(0x5E, 0x56, 0x4A);

const BORDER: f32 = 2.0;
const SHADOW: i8 = 6;

pub fn pixel(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name("pixel".into()))
}

pub fn mono(size: f32) -> FontId {
    FontId::new(size, FontFamily::Monospace)
}

pub fn body(size: f32) -> FontId {
    FontId::new(size, FontFamily::Proportional)
}

fn install_fonts(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    let add = |f: &mut FontDefinitions, name: &str, bytes: &'static [u8]| {
        f.font_data
            .insert(name.to_owned(), Arc::new(FontData::from_static(bytes)));
    };
    add(
        &mut fonts,
        "Inter",
        include_bytes!("../assets/fonts/Inter.ttf"),
    );
    add(
        &mut fonts,
        "PixelifySans",
        include_bytes!("../assets/fonts/PixelifySans.ttf"),
    );
    add(
        &mut fonts,
        "VT323",
        include_bytes!("../assets/fonts/VT323-Regular.ttf"),
    );

    let prop = fonts.families.entry(FontFamily::Proportional).or_default();
    prop.insert(0, "Inter".to_owned());
    let mono = fonts.families.entry(FontFamily::Monospace).or_default();
    mono.insert(0, "VT323".to_owned());
    // Pixel display face falls back to Inter for glyphs it lacks (→, °, —).
    let mut pixel = vec!["PixelifySans".to_owned(), "Inter".to_owned()];
    pixel.extend(
        fonts.families[&FontFamily::Proportional]
            .iter()
            .skip(1)
            .cloned(),
    );
    fonts
        .families
        .insert(FontFamily::Name("pixel".into()), pixel);
    ctx.set_fonts(fonts);
}

pub fn hard_shadow(px: i8, color: Color32) -> Shadow {
    Shadow {
        offset: [px, px],
        blur: 0,
        spread: 0,
        color,
    }
}

pub fn apply(ctx: &egui::Context, scale: f32) {
    install_fonts(ctx);
    ctx.set_pixels_per_point(ctx.native_pixels_per_point().unwrap_or(1.0) * scale);

    let mut style = (*ctx.global_style()).clone();
    style.text_styles = [
        (TextStyle::Heading, pixel(26.0)),
        (TextStyle::Body, body(13.5)),
        (TextStyle::Button, body(13.5)),
        (TextStyle::Small, mono(16.0)),
        (TextStyle::Monospace, mono(18.0)),
    ]
    .into();

    let ink = Stroke::new(1.5, INK);
    let mut v = egui::Visuals::light();
    v.override_text_color = Some(INK);
    v.weak_text_color = Some(INK_SOFT);
    v.hyperlink_color = COBALT;
    v.panel_fill = PARCHMENT;
    v.window_fill = BONE;
    v.window_stroke = Stroke::new(BORDER, INK);
    v.window_corner_radius = CornerRadius::ZERO;
    v.menu_corner_radius = CornerRadius::ZERO;
    v.window_shadow = hard_shadow(SHADOW, INK);
    v.popup_shadow = hard_shadow(4, INK);
    v.extreme_bg_color = BONE;
    v.faint_bg_color = PARCHMENT;
    v.code_bg_color = PARCHMENT;
    v.warn_fg_color = VERMILLION;
    v.error_fg_color = VERMILLION;
    v.selection.bg_fill = COBALT;
    v.selection.stroke = Stroke::new(1.5, BONE);
    v.slider_trailing_fill = true;
    v.handle_shape = egui::style::HandleShape::Rect { aspect_ratio: 0.55 };
    v.striped = false;
    v.indent_has_left_vline = false;

    let w = &mut v.widgets;
    for wv in [
        &mut w.noninteractive,
        &mut w.inactive,
        &mut w.hovered,
        &mut w.active,
        &mut w.open,
    ] {
        wv.corner_radius = CornerRadius::ZERO;
        wv.expansion = 0.0;
    }
    w.noninteractive.bg_fill = BONE;
    w.noninteractive.weak_bg_fill = BONE;
    w.noninteractive.bg_stroke = Stroke::new(1.0, TAN);
    w.noninteractive.fg_stroke = Stroke::new(1.0, INK);

    w.inactive.bg_fill = TAN;
    w.inactive.weak_bg_fill = BONE;
    w.inactive.bg_stroke = ink;
    w.inactive.fg_stroke = Stroke::new(1.5, INK);

    w.hovered.bg_fill = TAN;
    w.hovered.weak_bg_fill = PARCHMENT;
    w.hovered.bg_stroke = Stroke::new(2.0, INK);
    w.hovered.fg_stroke = Stroke::new(1.5, INK);

    w.active.bg_fill = COBALT;
    w.active.weak_bg_fill = COBALT;
    w.active.bg_stroke = Stroke::new(2.0, INK);
    w.active.fg_stroke = Stroke::new(1.5, BONE);

    w.open.bg_fill = PARCHMENT;
    w.open.weak_bg_fill = PARCHMENT;
    w.open.bg_stroke = Stroke::new(2.0, INK);
    w.open.fg_stroke = Stroke::new(1.5, INK);

    style.visuals = v;
    style.spacing.item_spacing = vec2(8.0, 7.0);
    style.spacing.button_padding = vec2(8.0, 4.0);
    style.spacing.interact_size.y = 24.0;
    style.spacing.slider_width = 150.0;
    style.spacing.combo_width = 200.0;
    style.spacing.icon_width = 14.0;
    ctx.set_global_style(style);
}

/// Bone card, ink border, hard offset shadow.
pub fn card(fill: Color32) -> Frame {
    Frame::new()
        .fill(fill)
        .stroke(Stroke::new(BORDER, INK))
        .shadow(hard_shadow(SHADOW, INK))
        .corner_radius(CornerRadius::ZERO)
        .inner_margin(Margin::same(14))
}

#[derive(Clone, Copy)]
pub enum Btn {
    Primary,
    Secondary,
}

/// Chunky button that sinks into its hard shadow when pressed.
pub fn button(ui: &mut Ui, text: &str, kind: Btn) -> Response {
    let (fill, fg) = match kind {
        Btn::Primary => (COBALT, BONE),
        Btn::Secondary => (BONE, INK),
    };
    let sh = 3.0;
    let galley = ui
        .painter()
        .layout_no_wrap(text.to_uppercase(), pixel(14.0), fg);
    let pad = vec2(12.0, 6.0);
    let body = galley.size() + pad * 2.0;
    let (rect, resp) = ui.allocate_exact_size(body + Vec2::splat(sh), Sense::click());
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    if ui.is_rect_visible(rect) {
        let base = egui::Rect::from_min_size(rect.min, body);
        let pressed = resp.is_pointer_button_down_on();
        let off = if pressed {
            sh
        } else if resp.hovered() {
            1.0
        } else {
            0.0
        };
        let p = ui.painter();
        if !pressed {
            p.rect_filled(base.translate(Vec2::splat(sh)), 0.0, INK);
        }
        let r = base.translate(Vec2::splat(off));
        p.rect_filled(r, 0.0, fill);
        p.rect_stroke(r, 0.0, Stroke::new(BORDER, INK), StrokeKind::Inside);
        p.galley(r.center() - galley.size() / 2.0, galley, fg);
    }
    resp
}

/// Square pixel checkbox: cobalt block when on.
pub fn check(ui: &mut Ui, value: &mut bool, label: &str) -> Response {
    let galley = ui
        .painter()
        .layout_no_wrap(label.to_owned(), body(13.5), INK);
    let bx = 16.0;
    let size = vec2(bx + 8.0 + galley.size().x, bx.max(galley.size().y));
    let (rect, mut resp) = ui.allocate_exact_size(size, Sense::click());
    if resp.clicked() {
        *value = !*value;
        resp.mark_changed();
    }
    if ui.is_rect_visible(rect) {
        let b = egui::Rect::from_min_size(
            egui::pos2(rect.min.x, rect.center().y - bx / 2.0),
            Vec2::splat(bx),
        );
        let p = ui.painter();
        p.rect_filled(b, 0.0, if *value { COBALT } else { BONE });
        p.rect_stroke(b, 0.0, Stroke::new(BORDER, INK), StrokeKind::Inside);
        if *value {
            p.rect_filled(b.shrink(5.0), 0.0, BONE);
        } else if resp.hovered() {
            p.rect_filled(b.shrink(5.0), 0.0, TAN);
        }
        p.galley(
            egui::pos2(b.max.x + 8.0, rect.center().y - galley.size().y / 2.0),
            galley,
            INK,
        );
    }
    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// Small tag chip ("● LIVE", "HOT DROP") with a 2px hard shadow.
pub fn tag(ui: &mut Ui, text: &str, fill: Color32, fg: Color32) -> Response {
    let galley = ui
        .painter()
        .layout_no_wrap(text.to_uppercase(), mono(16.0), fg);
    let pad = vec2(7.0, 1.0);
    let body = galley.size() + pad * 2.0;
    let (rect, resp) = ui.allocate_exact_size(body + Vec2::splat(2.0), Sense::hover());
    let r = egui::Rect::from_min_size(rect.min, body);
    let p = ui.painter();
    p.rect_filled(r.translate(Vec2::splat(2.0)), 0.0, INK);
    p.rect_filled(r, 0.0, fill);
    p.rect_stroke(r, 0.0, Stroke::new(1.5, INK), StrokeKind::Inside);
    p.galley(r.center() - galley.size() / 2.0, galley, fg);
    resp
}

/// Marginalia label: VT323, uppercase, soft ink.
pub fn margin_text(text: &str) -> RichText {
    RichText::new(text.to_uppercase())
        .font(mono(16.0))
        .color(INK_SOFT)
}

/// Numbered section ("02 // PICTURE") with a persisted open/closed state.
pub fn section(
    ui: &mut Ui,
    num: u32,
    title: &str,
    default_open: bool,
    add_body: impl FnOnce(&mut Ui),
) {
    let id = ui.make_persistent_id(("section", title));
    let mut open = ui.data_mut(|d| *d.get_persisted_mut_or(id, default_open));
    let width = ui.available_width();
    let (rect, resp) = ui.allocate_exact_size(vec2(width, 28.0), Sense::click());
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    if resp.clicked() {
        open = !open;
        ui.data_mut(|d| d.insert_persisted(id, open));
    }
    let p = ui.painter();
    if resp.hovered() {
        p.rect_filled(rect, 0.0, PARCHMENT);
    }
    let num_g = p.layout_no_wrap(format!("{num:02} //"), mono(18.0), COBALT);
    let title_g = p.layout_no_wrap(title.to_uppercase(), pixel(17.0), INK);
    let sign_g = p.layout_no_wrap(
        if open { "[-]" } else { "[+]" }.into(),
        mono(18.0),
        INK_SOFT,
    );
    let cy = rect.center().y;
    let nx = rect.min.x + 2.0;
    p.galley(
        egui::pos2(nx, cy - num_g.size().y / 2.0),
        num_g.clone(),
        COBALT,
    );
    p.galley(
        egui::pos2(nx + num_g.size().x + 8.0, cy - title_g.size().y / 2.0),
        title_g,
        INK,
    );
    p.galley(
        egui::pos2(
            rect.max.x - sign_g.size().x - 2.0,
            cy - sign_g.size().y / 2.0,
        ),
        sign_g,
        INK_SOFT,
    );
    // Ink rule under the header.
    p.line_segment(
        [
            egui::pos2(rect.min.x, rect.max.y),
            egui::pos2(rect.max.x, rect.max.y),
        ],
        Stroke::new(BORDER, INK),
    );
    if open {
        ui.add_space(6.0);
        add_body(ui);
        ui.add_space(10.0);
    } else {
        ui.add_space(4.0);
    }
}

/// Dashed rule, as used between editorial sections in the kit.
pub fn dashed_rule(ui: &mut Ui) {
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(vec2(width, 6.0), Sense::hover());
    let y = rect.center().y;
    let mut x = rect.min.x;
    while x < rect.max.x {
        let x2 = (x + 6.0).min(rect.max.x);
        ui.painter()
            .line_segment([egui::pos2(x, y), egui::pos2(x2, y)], Stroke::new(2.0, INK));
        x += 10.0;
    }
}

/// "CAP■VIEW." wordmark.
pub fn wordmark(ui: &mut Ui, size: f32) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        ui.label(RichText::new("CAP").font(pixel(size)).color(INK));
        let s = size * 0.42;
        let (r, _) = ui.allocate_exact_size(vec2(s + size * 0.16, size), Sense::hover());
        let sq = egui::Rect::from_center_size(
            egui::pos2(r.center().x, r.center().y + size * 0.04),
            Vec2::splat(s),
        );
        ui.painter().rect_filled(sq, 0.0, VERMILLION);
        ui.label(RichText::new("VIEW").font(pixel(size)).color(INK));
        ui.label(RichText::new(".").font(pixel(size)).color(VERMILLION));
    });
}
