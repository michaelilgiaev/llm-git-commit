"""Tests for digest.py -- the grounded commit line and the rewrite check."""

from __future__ import annotations

import digest

_DIR = """\
diff --git a/crates/ignore/src/dir.rs b/crates/ignore/src/dir.rs
--- a/crates/ignore/src/dir.rs
+++ b/crates/ignore/src/dir.rs
@@ -373,7 +373,7 @@ impl Ignore {
-        let git_type = if check_vcs_dir
+        let git_type = if (check_vcs_dir || self.inner.opts.git_exclude)
+        let without_required_git =
+            IgnoreBuilder::new().require_git(false).build();
"""

_GO = """\
diff --git a/screen.go b/screen.go
--- a/screen.go
+++ b/screen.go
+func isPlausibleWindowSize(width, height int) bool {
+	return width*height <= maxWindowCells
+}
diff --git a/tea.go b/tea.go
--- a/tea.go
+++ b/tea.go
+			if size, ok := msg.(WindowSizeMsg); ok && !isPlausibleWindowSize(size.Width, size.Height) {
+				continue
+			}
"""


def test_one_file_names_the_new_symbols():
    text = digest.line("", _DIR)
    assert "dir.rs" in text
    assert "git_exclude" in text
    assert ";" not in text


def test_several_files_are_one_behavior():
    # The call site skips the step when the new check fails. The helper name is not the subject.
    text = digest.line("", _GO)
    assert "skip" in text.casefold()
    assert "fail" in text.casefold()
    assert not text.startswith("Update ")


def test_stat_only_when_there_is_no_patch():
    stat = " README.md | 3 ++\n 1 file changed, 3 insertions(+)\n"
    assert digest.line(stat, "") == "Update README.md"


def test_ungrounded_rewrite_is_dropped():
    source = "use git_exclude in dir.rs"
    assert digest.choose("e2e8e9d; 34", source) == source
    assert digest.choose("", source) == source


def test_grounded_rewrite_is_kept():
    source = "use git_exclude in dir.rs"
    rewritten = "check git_exclude while scanning dir.rs"
    assert digest.choose(rewritten, source) == rewritten


def test_a_file_with_no_symbol_still_names_the_file():
    patch = """\
diff --git a/test_make_default_short_help.py b/test_make_default_short_help.py
--- a/test_make_default_short_help.py
+++ b/test_make_default_short_help.py
@@ -1,3 +1,4 @@ def test_short
+    assert abbreviation
"""
    text = digest.line("", patch)
    assert "test_make_default_short_help.py" in text
    assert text[0].isupper()


def test_a_long_test_name_does_not_hide_the_reason():
    patch = """\
diff --git a/signals_test.go b/signals_test.go
--- a/signals_test.go
+++ b/signals_test.go
+// TestHandleSignalsReturnsAfterContextCancelDuringPendingSend is a
+// regression test for the signal-goroutine deadlock on shutdown.
"""
    text = digest.line("", patch)
    assert "deadlock" in text
    assert text.startswith("Add a regression test")


def test_an_added_comment_says_what_changed():
    patch = """\
diff --git a/tty.go b/tty.go
--- a/tty.go
+++ b/tty.go
+// fallbackDimensions returns width and height, falling back to COLUMNS
+// or 80x24 when the terminal size is non-positive.
+func (p *Program) fallbackDimensions(width, height int) (int, int) {
"""
    text = digest.line("", patch)
    assert "fall back" in text.casefold()
    assert "non-positive" in text or "80x24" in text


def test_a_docstring_says_what_changed():
    patch = '''\
diff --git a/core.py b/core.py
--- a/core.py
+++ b/core.py
+    def get_help_record(self, ctx):
+        """Returns the argument's help row: its metavar and its help text.
+
+        Unlike Option.get_help_record, this never returns None.
+        """
+        return ("name", "")
'''
    text = digest.line("", patch)
    assert "help row" in text or "never return" in text.casefold()
    assert "return" in text.casefold()


def test_plain_words_are_not_named_as_the_change():
    patch = """\
diff --git a/utils.py b/utils.py
--- a/utils.py
+++ b/utils.py
+    words = help.partition("\\n\\n")[0].split()
+    if word.endswith(".") and break_on_hyphens:
"""
    text = digest.line("", patch)
    assert "break_on_hyphens" in text
    assert "partition" not in text
    assert "endswith" not in text


def test_plain_text_is_not_treated_as_a_digest():
    assert digest.choose("Add greeting helper", "the diff") == "Add greeting helper"


def test_multi_clause_rewrite_must_stay_split():
    source = "add isPlausibleWindowSize in screen.go; then use WindowSizeMsg in tea.go"
    collapsed = "add isPlausibleWindowSize and WindowSizeMsg in screen.go and tea.go"
    assert digest.choose(collapsed, source) == source


def test_rewrite_may_drop_helper_names():
    source = "add isPlausibleWindowSize in screen.go; then use WindowSizeMsg in tea.go"
    rewritten = (
        "ignore implausible sizes via isPlausibleWindowSize in screen.go; "
        "then update tea.go"
    )
    assert digest.choose(rewritten, source) == rewritten


def test_a_repeated_rephrase_is_dropped():
    import harness
    draft = "Update edit in termui.py"
    echo = (
        "Update warning message for editor command blocks; "
        "; Update warning message for editor command blocks;"
    )
    assert harness.pick(echo, draft) == draft


def test_a_repeated_private_rename_says_make_private():
    patch = """\
diff --git a/utils.py b/utils.py
--- a/utils.py
+++ b/utils.py
-def get_binary_stream(name):
+def _get_binary_stream(name):
-    return get_binary_stream("stdout")
+    return _get_binary_stream("stdout")
+    warnings.warn("old name", DeprecationWarning)
"""
    text = digest.line("", patch)
    assert "get_binary_stream" in text
    assert "private" in text.casefold()
    assert "deprecate" in text.casefold()


def test_a_swapped_symbol_says_instead():
    patch = """\
diff --git a/view.go b/view.go
--- a/view.go
+++ b/view.go
-    draw(view.Layer)
-    if view.Layer == nil {
+    draw(view.Content)
+    if view.Content == nil {
-    return view.Layer
+    return view.Content
"""
    text = digest.line("", patch)
    assert "Content" in text
    assert "Layer" in text
    assert "instead" in text.casefold() or text.startswith("Rename")


def test_a_class_named_path_is_not_called_pathlike():
    patch = """\
diff --git a/src/click/types.py b/src/click/types.py
--- a/src/click/types.py
+++ b/src/click/types.py
@@ -1,3 +1,4 @@ class Path(ParamType[str | bytes | os.PathLike[str]]):
-        is_dash = self.file_okay and rv in (b"-", "-")
+        dash = b"-" if isinstance(rv, bytes) else "-"
+        is_dash = self.file_okay and rv == dash
+        # avoid a BytesWarning under python -bb
"""
    text = digest.line("", patch)
    assert "PathLike" not in text
    assert "Path" in text


def test_close_before_unlink_names_the_reason():
    patch = """\
diff --git a/_termui_impl.py b/_termui_impl.py
--- a/_termui_impl.py
+++ b/_termui_impl.py
@@ -10,3 +10,6 @@ def _tempfilepager
+        # Closing here keeps a PermissionError from replacing the original
+        # exception, so Windows can still surface the paging error.
+        f.close()
+        os.unlink(f.name)
"""
    text = digest.line("", patch)
    assert "close" in text.casefold() or "unlink" in text.casefold()
    assert "windows" in text.casefold() or "permissionerror" in text.casefold()


def test_two_invented_symbols_are_dropped():
    source = "add isPlausibleWindowSize in screen.go"
    rewritten = "add isPlausibleWindowSize and fake_helper plus otherThing in screen.go"
    assert digest.choose(rewritten, source) == source


def test_a_regression_url_is_not_the_subject():
    patch = """\
diff --git a/tea.go b/tea.go
--- a/tea.go
+++ b/tea.go
-	return [...]string{"None"}[s]
+	// Values outside the known range return "Unknown".
+	return "Unknown"
diff --git a/tea_test.go b/tea_test.go
--- a/tea_test.go
+++ b/tea_test.go
+// TestString is a regression test for
+// https://example.com/issues/1: String panicked
"""
    text = digest.line("", patch)
    assert "https" not in text.casefold()
    assert "Unknown" in text


def test_an_error_check_is_not_a_nil_guard():
    patch = """\
diff --git a/renderer.go b/renderer.go
--- a/renderer.go
+++ b/renderer.go
@@ -1,3 +1,5 @@ func insertAbove
-	return err
+	if err != nil {
+		return fmt.Errorf("bubbletea: error writing insert above to the writer: %w", err)
+	}
"""
    text = digest.line("", patch)
    assert "nil err" not in text.casefold()
    assert "insert above" in text.casefold()


def test_a_behavior_draft_beats_an_update_rephrase():
    import harness
    draft = 'Return "Unknown" for values outside the known range instead of panicking'
    rewritten = "Update tea.go: Add a human-readable name for ProgressBarState"
    assert harness.pick(rewritten, draft, f"draft: {draft}\nfile: tea.go") == draft


def test_an_edited_function_keeps_never_returns():
    patch = '''\
diff --git a/core.py b/core.py
--- a/core.py
+++ b/core.py
@@ -1,2 +1,6 @@ def get_help_record(self, ctx):
+    """Returns the argument's help row.
+
+    Unlike Option.get_help_record, this never returns None.
+    """
     return ("name", "")
'''
    text = digest.line("", patch)
    assert "never return" in text.casefold()
    assert "none" in text.casefold()


def test_a_docstring_continuation_says_to_strip_ansi():
    patch = '''\
diff --git a/termui.py b/termui.py
--- a/termui.py
+++ b/termui.py
@@ -1,3 +1,8 @@ def _readline_prompt(func, text, err):
     """Call a prompt function.
+
+    The prompt is handed to func rather than written through echo, so it has to
+    strip ANSI color and style codes when the destination stream does not
+    support them.
     """
+    if should_strip_ansi(stream):
+        text = strip_ansi(text)
'''
    text = digest.line("", patch)
    assert "strip" in text.casefold()
    assert "ansi" in text.casefold()
    assert not text.startswith("Update ")


def test_map_clone_becomes_cloned():
    patch = """\
diff --git a/search.rs b/search.rs
--- a/search.rs
+++ b/search.rs
-        stats: sink.stats().map(|s| s.clone()),
+        stats: sink.stats().cloned(),
-        stats: sink.stats().map(|s| s.clone()),
+        stats: sink.stats().cloned(),
-        stats: sink.stats().map(|s| s.clone()),
+        stats: sink.stats().cloned(),
"""
    text = digest.line("", patch)
    assert "cloned" in text.casefold()
    assert not text.startswith("Update ")


def test_a_new_nil_guard_skips_the_call():
    patch = """\
diff --git a/tea.go b/tea.go
--- a/tea.go
+++ b/tea.go
@@ -1,3 +1,5 @@ func (p *Program) RestoreTerminal() error {
-	if err := p.initInputReader(false); err != nil {
+	if p.input != nil {
+		if err := p.initInputReader(false); err != nil {
 			return err
+		}
"""
    text = digest.line("", patch)
    assert "input reader" in text.casefold()
    assert "nil" in text.casefold()
    assert "RestoreTerminal" in text
    assert not text.startswith("Update ")


def test_a_prompt_so_clause_names_the_prompt():
    patch = '''\
diff --git a/termui.py b/termui.py
--- a/termui.py
+++ b/termui.py
@@ -1,3 +1,8 @@ def _readline_prompt(func, text, err):
     """Call a prompt function.
+
+    The prompt is handed to func rather than written through echo, so it has to
+    strip ANSI color and style codes when the destination stream does not
+    support them.
     """
+    if should_strip_ansi(stream):
+        text = strip_ansi(text)
'''
    text = digest.line("", patch)
    assert "from the prompt" in text.casefold()
    assert "strip" in text.casefold()


def test_a_single_method_names_its_type():
    patch = '''\
diff --git a/tea.go b/tea.go
--- a/tea.go
+++ b/tea.go
@@ -1,4 +1,6 @@ func (s ProgressBarState) String() string {
-	return names[s]
+	// Values outside the known range return "Unknown".
+	return "Unknown"
'''
    text = digest.line("", patch)
    assert "Unknown" in text
    assert "ProgressBarState.String" in text


def test_a_new_help_parameter_names_the_section():
    patch = '''\
diff --git a/core.py b/core.py
--- a/core.py
+++ b/core.py
@@ -1,3 +1,6 @@ class Argument(Parameter):
+        help: str | None = None,
+        """Added the ``help`` parameter."""
+        with formatter.section(_("Positional arguments")):
'''
    text = digest.line("", patch)
    assert "help parameter" in text.casefold()
    assert "Argument" in text
    assert "Positional arguments" in text
    assert "deprecat" not in text.casefold()


def test_a_parameter_named_deprecated_is_not_a_deprecation():
    patch = '''\
diff --git a/core.py b/core.py
--- a/core.py
+++ b/core.py
@@ -1,2 +1,5 @@ def format_arguments(self, ctx, formatter):
+    """Writes the arguments that have a help record into the formatter."""
+    deprecated = attrs.get("deprecated", False)
+    if deprecated:
+        label = "old"
'''
    text = digest.line("", patch)
    assert "argument" in text.casefold()
    assert "deprecat" not in text.casefold()


def test_slicing_a_file_name_names_it():
    patch = """\
diff --git a/pathutil.rs b/pathutil.rs
--- a/pathutil.rs
+++ b/pathutil.rs
@@ -1,4 +1,2 @@ pub(crate) fn file_name(path: &[u8]) {
-    let mut path = path.clone();
-    path.drain(..n);
+    let path = path[n..].to_vec();
@@ -5,4 +3,2 @@ pub(crate) fn file_name_ext(name: &[u8]) {
-    let mut name = name.clone();
-    name.drain(..n);
+    let name = name[n..].to_vec();
"""
    text = digest.line("", patch)
    assert "file name" in text.casefold()
    assert "extension" in text.casefold()
    assert not text.startswith("Update ")


def test_is_match_swapped_for_matches_all():
    patch = """\
diff --git a/lib.rs b/lib.rs
--- a/lib.rs
+++ b/lib.rs
@@ -1,3 +1,7 @@ fn matches_all(&self, path: &Path) -> bool {
-            if !strat.is_match(path) {
+            if !strat.matches_all(path) {
+                // If all the prefixes match, count every one.
+                fn matches_all(&self) -> bool { self.is_match(candidate) }
"""
    text = digest.line("", patch)
    assert "matches_all" in text
    assert "is_match" in text
    assert not text.startswith("Add ")


def test_a_forced_false_assumes_raw_mode():
    patch = """\
diff --git a/tea.go b/tea.go
--- a/tea.go
+++ b/tea.go
@@ -1,2 +1,4 @@ func (p *Program) Run() {
-	r.setOptimizations(p.ttyInput == nil)
+	// Working in cooked mode instead of raw mode breaks a terminal that is not a TTY.
+	mapNl := false
+	r.setOptimizations(mapNl)
"""
    text = digest.line("", patch)
    assert "newline" in text.casefold()
    assert not text.startswith("Update ")


def test_a_color_note_beats_a_chained_raise():
    patch = '''\
diff --git a/termui.py b/termui.py
--- a/termui.py
+++ b/termui.py
@@ -1,4 +1,6 @@ def _interpret_color(color):
-        raise TypeError("bad")
+        """All invalid color values raise ValueError. 256-color index 0 is not ignored."""
+        if 0 <= color:
+            raise ValueError("bad")
'''
    text = digest.line("", patch)
    assert "ValueError" in text
    assert "256-color" in text
    assert "TypeError" not in text


def test_a_generic_class_keeps_the_narrowing_reason():
    patch = """\
diff --git a/types.py b/types.py
--- a/types.py
+++ b/types.py
-class Path(ParamType):
+class Path(ParamType, Generic[T]):
+    T = TypeVar("T")
diff --git a/test_path.py b/test_path.py
--- a/test_path.py
+++ b/test_path.py
+    # path_type narrows the converted value's type for convert and prompt.
"""
    text = digest.line("", patch)
    assert "generic" in text.casefold()
    assert "narrows" in text.casefold()
    assert "path_type" in text


def test_a_redeclared_helper_is_not_an_extraction():
    patch = """\
diff --git a/pathutil.rs b/pathutil.rs
--- a/pathutil.rs
+++ b/pathutil.rs
-    fn imp(path: &Path) -> bool {
-        name.as_bytes().get(0) == Some(&b'.')
-    }
-    imp(path)
+    name.as_encoded_bytes().starts_with(b".")
-    fn imp(path: &Path) -> bool {
-        name.to_str().map(|s| s.starts_with(".")).unwrap_or(false)
-    }
+    #[cfg(unix)]
+    fn imp(path: &Path) -> Option<&OsStr> {
+        Some(path)
+    }
"""
    text = digest.line("", patch)
    assert "extract" not in text.casefold()
    assert "as_encoded_bytes" in text


def test_unlike_doc_says_the_hidden_case_still_returns_nothing():
    patch = '''\
diff --git a/core.py b/core.py
--- a/core.py
+++ b/core.py
+    def get_help_spec(self, ctx):
+        """Returns the spellings.
+        Unlike get_help_record, the spec is produced even when the option is hidden.
+        """
+        return joined
+    def get_help_record(self, ctx):
+        if self.hidden:
+            return None
+        return self.get_help_spec(ctx), help
'''
    text = digest.line("", patch)
    assert "get_help_spec" in text
    assert "hidden" in text.casefold()
    assert "reuse" in text.casefold()
    assert "get_help_record" in text


def test_pick_rejects_a_clause_dump():
    import harness
    draft = (
        "Add Option.get_help_spec to produce the spelling even when the option is hidden, "
        "and reuse it in get_help_record"
    )
    dumped = (
        "Add get_help_spec and get_help_record methods to core.py; "
        "return the left column; include spellings; unlike get_help_record, produce the spec"
    )
    assert harness.pick(dumped, draft, f"draft: {draft}") == draft


def test_pick_rejects_a_dropped_contrast():
    import harness
    draft = "Hit-test layers in the view callback instead of tea.LayerHitMsg"
    flipped = "Add LayerHitMsg struct to handle mouse events in view layers"
    assert harness.pick(flipped, draft, f"draft: {draft}") == draft


def test_pick_rejects_a_rephrase_that_drops_redraw():
    import harness
    draft = "Do not scroll a window due for a full redraw"
    weaker = "Fix stale grid offsets when window is not scrolled"
    assert harness.pick(weaker, draft, f"draft: {draft}") == draft


def test_fast_true_allows_the_named_function():
    patch = """\
diff --git a/eval.lua b/eval.lua
--- a/eval.lua
+++ b/eval.lua
@@ -1,4 +1,5 @@ M.funcs = {
     ]=],
+    fast = true,
     name = 'getenv',
     params = { { 'name', 'string' } },
"""
    text = digest.line("", patch)
    assert "getenv" in text
    assert "api-fast" in text


def test_pick_rejects_an_update_step_down():
    import harness
    draft = "Rename Layer to Content"
    weaker = "Update content representation in tea.go"
    assert harness.pick(weaker, draft, f"draft: {draft}\nfile: tea.go") == draft


def test_doc_examples_name_the_function_they_demonstrate():
    patch = """\
diff --git a/tea.go b/tea.go
--- a/tea.go
+++ b/tea.go
@@ -1,3 +1,10 @@ type Hittable interface {
+// This function accepts any type and tries to convert it to a string.
+//
+// Example:
+//
+//	layer := lipgloss.NewLayer("Hello")
+//	v := tea.NewView(canvas)
@@ -20,3 +27,8 @@ type KeyboardEnhancements struct {
+// Example:
+//
+//	v.SetContent(canvas)
"""
    text = digest.line("", patch)
    assert "NewView" in text
    assert "SetContent" in text
    assert "Update the docs" not in text


def test_a_helper_removed_from_two_files_names_both():
    patch = """\
diff --git a/examples/pager/main.go b/examples/pager/main.go
--- a/examples/pager/main.go
+++ b/examples/pager/main.go
@@ -1,6 +1,0 @@ func footerView() string {
-func max(a, b int) int {
-	if a > b {
-		return a
-	}
-	return b
-}
diff --git a/examples/package-manager/main.go b/examples/package-manager/main.go
--- a/examples/package-manager/main.go
+++ b/examples/package-manager/main.go
@@ -1,6 +1,0 @@ func download() {
-func max(a, b int) int {
-	if a > b {
-		return a
-	}
-	return b
-}
"""
    text = digest.line("", patch)
    assert "max" in text
    assert "pager" in text
    assert "package-manager" in text
    assert "main.go" not in text


def test_a_field_rename_names_the_type():
    patch = """\
diff --git a/tea.go b/tea.go
--- a/tea.go
+++ b/tea.go
@@ -1,8 +1,8 @@ type View struct {
-	Layer any
+	Content any
-	return v.Layer
+	return v.Content
-	old := v.Layer
+	old := v.Content
-	draw(v.Layer)
+	draw(v.Content)
"""
    text = digest.line("", patch)
    assert text.startswith("Rename View.Layer to View.Content")


def test_a_cannot_so_note_beats_also_check():
    patch = '''\
diff --git a/core.py b/core.py
--- a/core.py
+++ b/core.py
@@ -1,3 +1,5 @@ class Argument:
     def get_help_record(self, ctx):
-        return None
+        """An argument cannot be hidden, so an undocumented one still gets a row.
+        """
+        return (self.make_metavar(), self.help or "")
'''
    text = digest.line("", patch)
    assert "cannot be hidden" in text.casefold()
    assert "row" in text.casefold()
    assert "also check" not in text.casefold()


def test_new_type_names_in_a_list_name_the_list():
    patch = """\
diff --git a/overrides.js b/overrides.js
--- a/overrides.js
+++ b/overrides.js
@@ -1,3 +1,6 @@ const nodeTypesWithContentEnd = ([
+  "ClassProperty",
+  "PropertyDefinition",
+  "ClassPrivateProperty",
   "ExpressionStatement",
"""
    text = digest.line("", patch)
    assert text.startswith("Treat ClassProperty, PropertyDefinition, and ClassPrivateProperty")
    assert "content end" in text.casefold()


def test_a_third_person_note_beats_also_check():
    patch = """\
diff --git a/ui.c b/ui.c
--- a/ui.c
+++ b/ui.c
@@ -1,3 +1,6 @@ void ui_refresh(void)
+// Sends a restart event to all UIs when the address changes.
+if (via_addr && chan->addr) {
+  remote_ui_restart(addr, reuse_addr);
+}
"""
    text = digest.line("", patch)
    assert "restart" in text.casefold()
    assert "also check" not in text.casefold()


def test_a_help_page_does_not_become_the_second_clause():
    patch = """\
diff --git a/ui.c b/ui.c
--- a/ui.c
+++ b/ui.c
@@ -1,3 +1,4 @@ void ui_refresh(void)
-  if (ready)
+  if (ready || via_addr)
+    remote_ui_restart(addr, reuse_addr);
diff --git a/api.txt b/api.txt
--- a/api.txt
+++ b/api.txt
+The old server is about to exit and close its channel before the client attaches.
"""
    text = digest.line("", patch)
    assert "document that" not in text.casefold()


def test_use_the_call_that_does_not_trim():
    patch = """\
diff --git a/printer.js b/printer.js
--- a/printer.js
+++ b/printer.js
@@ -1,3 +1,5 @@ function print(node)
-  return hardline;
+  // hardline trims the trailing whitespace of the block, use literalline
+  return literalline;
"""
    text = digest.line("", patch)
    assert "literalline" in text
    assert "not trimmed" in text.casefold()


def test_a_moved_c_function_names_its_caller():
    patch = """\
diff --git a/eval.c b/eval.c
--- a/eval.c
+++ b/eval.c
@@ -1,8 +1,14 @@ static int eval4(char **arg, typval_T *rettv)
-  int len = 2;
-  switch (p[0]) {
-  case '=':
-    type = EXPR_EQUAL;
-    break;
-  }
+static exprtype_T get_compare_type(const char *p, int *len)
+{
+  int len = 2;
+  switch (p[0]) {
+  case '=':
+    type = EXPR_EQUAL;
+    break;
+  }
+  return type;
+}
+static int eval4(char **arg, typval_T *rettv)
+{
+  exprtype_T type = get_compare_type(p, &len);
"""
    text = digest.line("", patch)
    assert text.startswith("Extract get_compare_type from eval4")


def test_an_include_does_not_swallow_the_reason():
    patch = """\
diff --git a/ui.c b/ui.c
--- a/ui.c
+++ b/ui.c
@@ -1,3 +1,7 @@ void remote_ui_restart(String listen_addr)
+#include "nvim/msgpack_rpc/server.h"
+/// @param reuse_addr The new server reused the listen address, so UIs can connect via their stored addr.
+bool via_addr = reuse_addr && addr;
+if (via_addr) {
+  push_call(ui, "restart", args);
+}
"""
    text = digest.line("", patch)
    assert "also check" not in text.casefold()
    assert "connect" in text.casefold()
    assert "include" not in text.casefold()


def test_implements_names_the_interface():
    patch = """\
diff --git a/db.go b/db.go
--- a/db.go
+++ b/db.go
@@ -1,3 +1,6 @@ func (a *appender) Commit() error {
+// GetRef implements storage.GetRef.
+func (a *appenderBase) GetRef(lset labels.Labels) (storage.SeriesRef, labels.Labels) {
+    return 0, labels.EmptyLabels()
+}
"""
    text = digest.line("", patch)
    assert text == "Implement storage.GetRef"


def test_two_type_names_name_the_predicate():
    patch = """\
diff --git a/handle.js b/handle.js
--- a/handle.js
+++ b/handle.js
@@ -1,4 +1,6 @@ const isFunctionLikeNode = createTypeCheckFunction([
   "ObjectMethod",
+  "ComponentDeclaration",
+  "HookDeclaration",
 ]);
"""
    text = digest.line("", patch)
    assert "ComponentDeclaration" in text
    assert "HookDeclaration" in text
    assert "function like" in text.casefold()
    assert not text.startswith("Update ")


def test_a_lua_doc_says_what_the_new_function_parses():
    patch = """\
diff --git a/uri.lua b/uri.lua
--- a/uri.lua
+++ b/uri.lua
@@ -1,2 +1,6 @@
+--- Parses a URI into its RFC 3986 components, without decoding them.
+---@return vim.Uri?
+function M.parse(uri)
+  return nil
+end
"""
    text = digest.line("", patch)
    assert text.startswith("Parse a URI into its RFC 3986 components")
    assert "also check" not in text.casefold()


def test_deprecated_wrappers_are_named_once():
    patch = """\
diff --git a/deprecated.c b/deprecated.c
--- a/deprecated.c
+++ b/deprecated.c
@@ -1,2 +1,8 @@
+/// "rpcnotify()" function (DEPRECATED)
+void f_rpcnotify(typval_T *argvars, typval_T *rettv)
+{
+}
+/// "rpcrequest()" function (DEPRECATED)
+void f_rpcrequest(typval_T *argvars, typval_T *rettv)
+{
+}
"""
    text = digest.line("", patch)
    assert text == "Add the deprecated rpcnotify and rpcrequest functions"


def test_a_colon_note_says_to_query_the_spanned_rows():
    patch = """\
diff --git a/highlighter.lua b/highlighter.lua
--- a/highlighter.lua
+++ b/highlighter.lua
@@ -1,2 +1,8 @@ function M:on_win()
+  -- This row's query need not return the capture of a `conceal_lines` mark that spans other rows
+  -- (e.g. moved by `#offset!`, or shrunk by an edit there): query its rows.
+  local erow = mark.end_row or row
+  self.tree:parse({ row, erow + 1 })
"""
    text = digest.line("", patch)
    assert text.startswith("Query its rows, because this row's query")
    assert "conceal_lines" in text
    assert "offset" not in text


def test_pick_keeps_a_faithful_second_clause():
    import harness
    draft = "Restore tab stops if we have tab optimizations enabled"
    better = "Restore tab stops if hard tabs are enabled; otherwise, set them to -1"
    assert harness.pick(better, draft, f"draft: {draft}") == better


def test_a_new_lua_function_returns_the_documented_value():
    patch = """\
diff --git a/util.lua b/util.lua
--- a/util.lua
+++ b/util.lua
@@ -1,1 +1,5 @@
+--- Returns the hostname for the statusline, if Nvim is running in a SSH session.
+--- @return string
+function M.remote_host()
+  return ''
+end
"""
    text = digest.line("", patch)
    assert text.startswith("Return the hostname for the statusline when Nvim is running")
    assert "remote_host" not in text


def test_a_comment_that_grows_by_or_names_the_new_case():
    patch = """\
diff --git a/editorconfig.lua b/editorconfig.lua
--- a/editorconfig.lua
+++ b/editorconfig.lua
@@ -1,4 +1,6 @@ local function glob2regpat(glob)
---- Modified version of glob2regpat() that does not match path separators on `*`.
+--- Modified version of glob2regpat() that does not match path separators on `*` or `?`.
+local qplaceholder = '@@QUESTION@@'
+return (regpat:gsub(placeholder, '[^/]*'):gsub(qplaceholder, '[^/]'))
"""
    text = digest.line("", patch)
    assert "path separators" in text.casefold()
    assert "?" in text
    assert "also check" not in text.casefold()


def test_a_call_moved_into_pcall_uses_the_existing_error_path():
    patch = """\
diff --git a/editorconfig.lua b/editorconfig.lua
--- a/editorconfig.lua
+++ b/editorconfig.lua
@@ -1,6 +1,8 @@ local function parse(filepath, dir)
-        local ok, regpat = pcall(glob2regpat, glob)
-          pat = vim.regex(regpat)
+        local ok, regex = pcall(function()
+          return vim.regex(glob2regpat(glob))
+        end)
+          pat = regex
"""
    text = digest.line("", patch)
    assert "vim.regex" in text
    assert "pcall" in text
    assert "also check" not in text.casefold()


def test_a_star_literal_branches_on_the_new_field():
    patch = """\
diff --git a/module.js b/module.js
--- a/module.js
+++ b/module.js
@@ -1,5 +1,9 @@ function printModuleSpecifier(path, options, print) {
-    left = "*";
+  if (node.exportsFilter) {
+    left = printObject(path, options, print);
+  } else {
+    left = "*";
+  }
"""
    text = digest.line("", patch)
    assert "printObject" in text
    assert "exportsFilter" in text
    assert "*" in text


def test_a_changelog_tweak_does_not_replace_the_code_change():
    patch = """\
diff --git a/changelog/20427.md b/changelog/20427.md
--- a/changelog/20427.md
+++ b/changelog/20427.md
@@ -1,2 +1,2 @@
-#### Add support for "Structs", "Import Sync", and "Deferred re-exports" proposal (#20427, #20430, #20430 by @fisker)
+#### Add support for "Structs", "Import Sync", and "Deferred re-exports" proposal (#20427, #20430, #20430, #20438 by @fisker)
diff --git a/module.js b/module.js
--- a/module.js
+++ b/module.js
@@ -1,5 +1,9 @@ function printModuleSpecifier() {
-    left = "*";
+  if (node.exportsFilter) {
+    left = printObject(path, options, print);
+  } else {
+    left = "*";
+  }
"""
    text = digest.line("", patch)
    assert "Structs" not in text
    assert "printObject" in text


def test_unescape_wraps_both_arguments():
    patch = """\
diff --git a/openmetricsparse.go b/openmetricsparse.go
--- a/openmetricsparse.go
+++ b/openmetricsparse.go
@@ -1,3 +1,3 @@ func (p *Parser) Exemplar(e *exemplar.Exemplar) bool {
-    p.builder.Add(s[a:b], s[c:d])
+    p.builder.Add(unreplace(s[a:b]), unreplace(s[c:d]))
"""
    text = digest.line("", patch)
    assert "Unescape" in text
    assert "Add" in text
    assert "also check" not in text.casefold()


def test_one_test_title_says_the_interrupt():
    patch = """\
diff --git a/async.lua b/async.lua
--- a/async.lua
+++ b/async.lua
@@ -1,3 +1,6 @@ function Task:wait(timeout, predicate)
+  local done, code = vim.wait(timeout, predicate)
+  if code == -2 then
+    error('interrupted', 0)
diff --git a/test/functional/lua/async_spec.lua b/test/functional/lua/async_spec.lua
--- a/test/functional/lua/async_spec.lua
+++ b/test/functional/lua/async_spec.lua
@@ -1,1 +1,2 @@
+it('CTRL-C interrupts a sync wait', function()
+end)
"""
    text = digest.line("", patch)
    assert "interruptible by CTRL-C" in text
    assert "also check" not in text.casefold()


def test_shared_crash_title_is_one_line():
    patch = """\
diff --git a/test/functional/crash_spec.lua b/test/functional/crash_spec.lua
--- a/test/functional/crash_spec.lua
+++ b/test/functional/crash_spec.lua
@@ -1,1 +1,4 @@
+it('no crash when ending Visual mode closes the window being closed', function()
+end)
+it('no crash when ending Visual mode closes the window to switch to', function()
+end)
"""
    text = digest.line("", patch)
    assert "regression test" in text.casefold()
    assert "Visual mode" in text
    assert "Avoid a crash" not in text


def test_count_title_drops_the_group_name():
    patch = """\
diff --git a/src/nvim/normal.c b/src/nvim/normal.c
--- a/src/nvim/normal.c
+++ b/src/nvim/normal.c
@@ -1,2 +1,2 @@ void exec_stuffed(cmdarg_T *cap)
-  ca.opcount = count;
+  ca.opcount = 0;
diff --git a/test/functional/cmdatom_spec.lua b/test/functional/cmdatom_spec.lua
--- a/test/functional/cmdatom_spec.lua
+++ b/test/functional/cmdatom_spec.lua
@@ -1,1 +1,2 @@
+it('exec_stuffed: count before a register is applied once #42468', function()
+end)
"""
    text = digest.line("", patch)
    assert "count before a register" in text
    assert "exec_stuffed" not in text


def test_drain_comment_is_the_subject():
    patch = """\
diff --git a/tsdb/fileutil/file.go b/tsdb/fileutil/file.go
--- a/tsdb/fileutil/file.go
+++ b/tsdb/fileutil/file.go
@@ -1,3 +1,6 @@ func (g *Group) refresh() {
+// Drain watcher events and errors in a dedicated goroutine for the entire lifetime of the watcher.
+go g.drain()
 }
"""
    text = digest.line("", patch)
    assert text.startswith("Drain watcher")
    assert "Update" not in text


def test_a_code_tail_is_not_a_second_clause():
    import harness
    text = harness._drop_code_tail(
        "Stop the truncate loop before taking the lock; db.mtx.Lock() | defer db.mtx.Unlock()"
    )
    assert "Lock" not in text
    assert text.startswith("Stop the truncate")


def test_character_class_comment_names_both_spellings():
    patch = """\
diff --git a/editorconfig.lua b/editorconfig.lua
--- a/editorconfig.lua
+++ b/editorconfig.lua
@@ -1,2 +1,4 @@ local function glob_to_vim(glob)
+-- EditorConfig negates a character class with "[!seq]", Vim with "[^seq]".
+glob = glob:gsub('[!]', '[^]')
"""
    text = digest.line("", patch)
    assert "[!seq]" in text
    assert "[^seq]" in text
    assert "EditorConfig" in text


def test_getenv_moves_to_the_env_table():
    patch = """\
diff --git a/log.lua b/log.lua
--- a/log.lua
+++ b/log.lua
@@ -1,2 +1,1 @@ function header(info)
-  local vimruntime = vim.uv.os_getenv('VIMRUNTIME')
-  local rt_path = vimruntime and vim.fs.relpath(vimruntime, src)
+  local rt_path = vim.env.VIMRUNTIME and vim.fs.relpath(vim.env.VIMRUNTIME, src)
"""
    text = digest.line("", patch)
    assert "VIMRUNTIME" in text
    assert "getenv" in text.casefold()


def test_a_window_due_for_redraw_is_not_scrolled():
    patch = """\
diff --git a/move.c b/move.c
--- a/move.c
+++ b/move.c
@@ -1,2 +1,4 @@ void win_scroll_lines(win_T *wp) {
-  if (wp->w_grid.target) {
+  // A window due for a full redraw is not scrolled: its grid offsets may be stale.
+  if (wp->w_grid.target
+      && wp->w_redr_type < UPD_NOT_VALID) {
"""
    text = digest.line("", patch)
    assert "redraw" in text.casefold()
    assert "scroll" in text.casefold()
    assert "also check" not in text.casefold()


def test_shift_constant_names_both_widths():
    patch = """\
diff --git a/linode.go b/linode.go
--- a/linode.go
+++ b/linode.go
@@ -1,1 +1,1 @@ func refreshData() {
-  label = strconv.FormatInt(int64(instance.Specs.Transfer)<<20, 10)
+  label = strconv.FormatInt(int64(instance.Specs.Transfer)<<30, 10)
"""
    text = digest.line("", patch)
    assert "GiB" in text and "MiB" in text


def test_prerelease_returns_a_version_instead_of_nil():
    patch = """\
diff --git a/ssh.lua b/ssh.lua
--- a/ssh.lua
+++ b/ssh.lua
@@ -1,2 +1,2 @@ function get_release(ver, base)
-  return dir, release_url, not ver.prerelease and base or nil
+  return dir, release_url, ver.prerelease and ('%s-%s'):format(base, ver.prerelease) or base
"""
    text = digest.line("", patch)
    assert "prerelease" in text.casefold()
    assert "nil" in text.casefold()
    assert "update" not in text.casefold()


def test_rebuilding_a_string_in_a_loop_is_one_pass():
    patch = """\
diff --git a/strip-comments.js b/strip-comments.js
--- a/strip-comments.js
+++ b/strip-comments.js
@@ -1,8 +1,9 @@ function stripCommentsFromText(text, comments) {
-    text =
-      text.slice(0, start) +
-      replaceNonLineBreaksWithSpace(text.slice(start, end)) +
-      text.slice(end);
+    result +=
+      text.slice(previousEnd, start) +
+      replaceNonLineBreaksWithSpace(text.slice(start, end));
+    previousEnd = end;
"""
    text = digest.line("", patch)
    assert "one pass" in text.casefold()
    assert "update" not in text.casefold()


def test_pick_rejects_an_update_that_drops_the_operator():
    import harness
    draft = "Add bitwise AND, OR, and XOR operators"
    weaker = "Update test for bitwise operators"
    assert harness.pick(weaker, draft, f"draft: {draft}") == draft


def test_a_reordered_struct_keeps_the_fields():
    patch = """\
diff --git a/scrape/scrape.go b/scrape/scrape.go
--- a/scrape/scrape.go
+++ b/scrape/scrape.go
@@ -1,3 +1,3 @@ type scraper struct {
-	offsetSeed uint64
-	metrics    *scrapeMetrics
+	metrics    *scrapeMetrics
+	offsetSeed uint64
diff --git a/tsdb/chunkenc/xor.go b/tsdb/chunkenc/xor.go
--- a/tsdb/chunkenc/xor.go
+++ b/tsdb/chunkenc/xor.go
@@ -1,2 +1,2 @@ type iterator struct {
-	t   int64
+	t int64
 	val float64
"""
    text = digest.line("", patch)
    assert "reorder" in text.casefold()
    assert "scrape" in text.casefold()


def test_a_must_not_comment_is_the_subject():
    patch = """\
diff --git a/django/utils/cache.py b/django/utils/cache.py
--- a/django/utils/cache.py
+++ b/django/utils/cache.py
@@ -10,6 +10,9 @@ def _generate_cache_key(request, headerlist):
+            # An absent field must not share a cache key with a field that is
+            # present but empty. Emit a marker that cannot collide.
+            ctx.update(b"-,")
"""
    text = digest.line("", patch)
    assert text.startswith("Do not let an absent field share")
    assert "empty" in text
    assert not text.startswith("Update ")


def test_a_rather_than_comment_names_the_behavior():
    patch = """\
diff --git a/utils/index.js b/utils/index.js
--- a/utils/index.js
+++ b/utils/index.js
@@ -1,4 +1,8 @@ function format(match, arg) {
-        return arg;
+        if (escaped || args.length === 0) {
+          // Specifiers with no argument left stay literal, like the console, rather than becoming undefined.
+          return match;
+        }
"""
    text = digest.line("", patch)
    assert "rather than" in text.casefold()
    assert "literal" in text.casefold()
    assert not text.startswith("Update ")
    assert "update" not in text.casefold()


def test_a_dollar_in_a_character_class_keeps_the_comment():
    patch = """\
diff --git a/scanner.go b/scanner.go
--- a/scanner.go
+++ b/scanner.go
@@ -1,3 +1,6 @@ func charAndSize(s string) {
+    // isASCIIIdentifierPart reports whether a byte is [A-Za-z0-9_$], with one load instead of a chain of range checks.
+    if isASCIIIdentifierPart(b) {
+        return 1
+    }
"""
    text = digest.line("", patch)
    assert "instead of" in text.casefold()
    assert "range" in text.casefold()
    assert not text.startswith("Update ")


def test_one_comment_replaces_an_inventory_line():
    patch = """\
diff --git a/query_utils.py b/query_utils.py
--- a/query_utils.py
+++ b/query_utils.py
@@ -1,4 +1,7 @@ def refresh(self):
+        # FETCH_PEERS may not load the field if the instance's row was deleted after the queryset was evaluated.
+        if self.__dict__.get(field) is missing:
+            return
"""
    text = digest.line("", patch)
    assert "deleted" in text.casefold()
    assert not text.startswith("Also check")
    assert not text.startswith("Update ")


def test_a_feature_comment_beats_a_return_contract():
    patch = """\
diff --git a/slowlog.c b/slowlog.c
--- a/slowlog.c
+++ b/slowlog.c
@@ -1,6 +1,10 @@ void track(void) {
+    /* Off-CPU tracking samples the main thread CPU time once per wakeup, to report
+     * commands in the slowlog that were starved for at least 10 milliseconds. */
+    /* Return 1 if a command that took duration microseconds would be logged. */
+    if (sample_off_cpu())
+        slowlogPush(c);
"""
    text = digest.line("", patch)
    assert "slowlog" in text.casefold() or "off-cpu" in text.casefold()
    assert not text.startswith("Return 1")


def test_orelseget_defers_the_fallback():
    patch = """\
diff --git a/StreamsGroup.java b/StreamsGroup.java
--- a/StreamsGroup.java
+++ b/StreamsGroup.java
@@ -1,4 +1,4 @@ class StreamsGroup {
-                    .orElse(
-                        topology.get(committedOffset)
+                    .orElseGet(
+                        () -> topology.get(committedOffset)
                     )
"""
    text = digest.line("", patch)
    assert text.startswith("Defer topology.get")
    assert "absent" in text.casefold()
    assert "orelse" not in text.casefold()


def test_parentheses_stay_on_a_type_argument():
    patch = """\
diff --git a/needs-parentheses.js b/needs-parentheses.js
--- a/needs-parentheses.js
+++ b/needs-parentheses.js
@@ -1,2 +1,3 @@ function needsParens(path) {
+    case "TSInstantiationExpression":
     return false
diff --git a/operand.ts b/operand.ts
--- a/operand.ts
+++ b/operand.ts
@@ -1,1 +1,2 @@
+x = (a + b)<T>;
"""
    text = digest.line("", patch)
    assert "(a + b)<T>" in text
    assert "parenthesize a tsinstantiation" not in text.casefold()


def test_a_permanently_enabled_flag_is_the_subject():
    patch = """\
diff --git a/main.go b/main.go
--- a/main.go
+++ b/main.go
@@ -1,3 +1,3 @@ func setFeatures(o string) {
-			case "promql-unquoted-utf8-names":
-				c.parserOpts.EnableUnquotedUTF8Names = true
+			case "promql-unquoted-utf8-names":
+				logger.Warn("This option for --enable-feature is now permanently enabled and therefore a no-op.", "option", o)
"""
    text = digest.line("", patch)
    assert "promql-unquoted-utf8-names" in text
    assert "no-op" in text.casefold()


def test_pick_rejects_a_code_tail():
    import harness
    draft = "Stop the truncate loop before taking db.mtx, since truncate needs it"
    raw = draft + "; db.mtx.Lock() | defer db.mtx.Unlock()"
    assert harness.pick(raw, draft) == draft


def test_pick_rejects_a_rephrase_that_drops_when():
    import harness
    draft = (
        "Stop the file watch when the watched file's parent directory is deleted, "
        "to avoid endless EPERM errors"
    )
    raw = "Stop handle to prevent endless EPERM errors"
    assert harness.pick(raw, draft) == draft
    garbled = "Stop handling when preventing endless EPERM errors"
    assert harness.pick(garbled, draft) == draft


def test_a_new_config_key_is_the_subject():
    patch = """\
diff --git a/redis.conf b/redis.conf
--- a/redis.conf
+++ b/redis.conf
+# Off-CPU tracking samples the main thread's CPU time once per event-loop
+# wakeup, to report commands (in the slowlog) that were off-CPU for at least
+# 10 milliseconds.
+off-cpu-tracking yes
diff --git a/src/config.c b/src/config.c
--- a/src/config.c
+++ b/src/config.c
+createBoolConfig("off-cpu-tracking", NULL, MODIFIABLE_CONFIG, server.off_cpu_tracking, 1, NULL, NULL),
"""
    text = digest.line("", patch)
    assert text.startswith("Add off-cpu-tracking to sample")
    assert "slowlog" in text
    assert "10 milliseconds" in text


def test_progress_sequences_end_with_st():
    patch = """\
diff --git a/lib/std/Progress.zig b/lib/std/Progress.zig
--- a/lib/std/Progress.zig
+++ b/lib/std/Progress.zig
-const progress_remove = "\\x1b]9;4;0\\x07";
-const progress_normal = "\\x1b]9;4;1;{d}\\x07";
-const progress_error = "\\x1b]9;4;2;{d}\\x07";
-const progress_pulsing = "\\x1b]9;4;3\\x07";
+const progress_remove = "\\x1b]9;4;0\\x1b\\\\";
+const progress_normal = "\\x1b]9;4;1;{d}\\x1b\\\\";
+const progress_error = "\\x1b]9;4;2;{d}\\x1b\\\\";
+const progress_pulsing = "\\x1b]9;4;3\\x1b\\\\";
"""
    text = digest.line("", patch)
    assert "ST instead of BEL" in text
    assert "OSC 9;4" in text


def test_override_annotations_are_the_subject():
    patch = """\
diff --git a/group/A.java b/group/A.java
--- a/group/A.java
+++ b/group/A.java
+    @Override
+    @Override
diff --git a/group/B.java b/group/B.java
--- a/group/B.java
+++ b/group/B.java
+    @Override
+    @Override
"""
    text = digest.line("", patch)
    assert text == "Mark overridden methods with @Override"


def test_a_later_registration_replaces_instead_of_rejecting():
    patch = """\
diff --git a/src/functions.c b/src/functions.c
--- a/src/functions.c
+++ b/src/functions.c
-    if (dictAdd(lib->functions, key, fn) != DICT_OK) {
-        err = sdsnew("Function already exists in the library");
-        goto error;
-    }
+    /* A later registration replaces an earlier function with the same name
+     * (case insensitive), freeing the earlier function and its metadata. */
+    old = dictFind(lib->functions, key);
+    if (old) functionFree(old);
+    dictAdd(lib->functions, key, fn);
diff --git a/src/dict.c b/src/dict.c
--- a/src/dict.c
+++ b/src/dict.c
-uint64_t dictHash(const void *key) { return siphash(key); }
-int dictCmp(const void *a, const void *b) { return memcmp(a, b); }
+uint64_t dictHash(const void *key) { return siphashCase(key); }
+int dictCmp(const void *a, const void *b) { return memcmpCase(a, b); }
"""
    text = digest.line("", patch)
    assert text.startswith("Replace an earlier function")
    assert "case-insensitively" in text
    assert "instead of rejecting" in text


def test_indented_input_type_is_the_subject():
    patch = """\
diff --git a/django/forms/widgets.py b/django/forms/widgets.py
--- a/django/forms/widgets.py
+++ b/django/forms/widgets.py
 class Textarea(Widget):
-    template_name = "django/forms/widgets/textarea.html"
+    template_name = "django/forms/widgets/textarea.html"
+    input_type = "textarea"
diff --git a/django/forms/templates/django/forms/widgets/textarea.html b/django/forms/templates/django/forms/widgets/textarea.html
--- a/django/forms/templates/django/forms/widgets/textarea.html
+++ b/django/forms/templates/django/forms/widgets/textarea.html
-<textarea name="{{ widget.name }}"{% include "django/forms/widgets/attrs.html" %}>
+<textarea name="{{ widget.name }}"{% if widget.is_hidden %} hidden{% endif %}{% include "django/forms/widgets/attrs.html" %}>
"""
    text = digest.line("", patch)
    assert text.startswith("Add input_type to Textarea")
    assert "hidden" in text


def test_short_clobber_flag_is_the_subject():
    old = ": .{ .memory = true, .icc = true, .o0 = true });"
    new = ": .{ .memory = true, .xcc = true, .o0 = true });"
    body = ""
    for index in range(4):
        body += f"""\
@@ pub fn syscall{index}() u64 {{
-{old}
+{new}
"""
    patch = "diff --git a/lib/std/os/linux/sparc64.zig b/lib/std/os/linux/sparc64.zig\n--- a/lib/std/os/linux/sparc64.zig\n+++ b/lib/std/os/linux/sparc64.zig\n" + body
    text = digest.line("", patch)
    assert "xcc instead of icc" in text
    assert "syscall" in text


def test_null_key_names_the_signature_not_the_hunk_header():
    patch = """\
diff --git a/src/module.c b/src/module.c
--- a/src/module.c
+++ b/src/module.c
@@ RedisModuleString *RM_ListGet(RedisModuleKey *key, long index) {
 int RM_ListSet(RedisModuleKey *key, long index, RedisModuleString *value) {
-    if (!value) {
+    if (!key || !value) {
         return REDISMODULE_ERR;
     }
@@ int RM_ListSet(RedisModuleKey *key, long index, RedisModuleString *value) {
 int RM_ListInsert(RedisModuleKey *key, long index, RedisModuleString *value) {
-    if (!value) {
+    if (!key || !value) {
         return REDISMODULE_ERR;
     }
"""
    text = digest.line("", patch)
    assert "RM_ListSet" in text and "RM_ListInsert" in text
    assert "RM_ListGet" not in text


def test_forgotten_node_keeps_the_cursor():
    patch = """\
diff --git a/src/cluster_legacy.c b/src/cluster_legacy.c
--- a/src/cluster_legacy.c
+++ b/src/cluster_legacy.c
@@ void clusterProcessPingExtensions(clusterMsg *hdr, clusterLink *link) {
+        clusterMsgPingExt *next = getNextPingExt(ext);
+        /* Compute the cursor to the next extension before running any handler. */
+        if (n == sender) {
+            serverLog(LL_WARNING, "Ignoring FORGOTTEN_NODE for active sender");
+        }
-        ext = getNextPingExt(ext);
+        ext = next;
"""
    text = digest.line("", patch)
    assert "FORGOTTEN_NODE" in text
    assert "active sender" in text
    assert "next extension" in text


def test_escaped_specifier_is_not_an_argument():
    patch = """\
diff --git a/packages/utils/index.js b/packages/utils/index.js
--- a/packages/utils/index.js
+++ b/packages/utils/index.js
@@ export function formatConsoleArgumentsToSingleString(args) {
+        if (escaped || args.length === 0) {
+          // Escaped specifiers don't take an argument.
+          return match;
+        }
diff --git a/src/__tests__/utils-test.js b/src/__tests__/utils-test.js
--- a/src/__tests__/utils-test.js
+++ b/src/__tests__/utils-test.js
+    it('should not consume arguments for escaped specifiers', () => {});
"""
    text = digest.line("", patch)
    assert "escaped" in text
    assert "formatConsoleArgumentsToSingleString" in text
    assert "stop consuming" not in text


def test_one_byte_slice_is_dropped():
    patch = """\
diff --git a/compiler/infer_types.rs b/compiler/infer_types.rs
--- a/compiler/infer_types.rs
+++ b/compiler/infer_types.rs
@@ fn is_ref_like_name(object_name: &str) -> bool {
-            && object_name[..1]
+            && object_name
                 .chars()
                 .next()
"""
    text = digest.line("", patch)
    assert "one-byte slice" in text
    assert "is_ref_like_name" in text
    assert "whole name" in text
    assert "correctly" not in text


def test_non_ascii_fixture_names_the_slice_purpose():
    patch = """\
diff --git a/compiler/infer_types.rs b/compiler/infer_types.rs
--- a/compiler/infer_types.rs
+++ b/compiler/infer_types.rs
@@ fn is_ref_like_name(object_name: &str) -> bool {
-            && object_name[..1]
+            && object_name
                 .chars()
                 .next()
diff --git a/fixtures/repro-ref-like-name-non-ascii.js b/fixtures/repro-ref-like-name-non-ascii.js
--- a/fixtures/repro-ref-like-name-non-ascii.js
+++ b/fixtures/repro-ref-like-name-non-ascii.js
+const ref = useRef(0);
"""
    text = digest.line("", patch)
    assert "non-ASCII" in text
    assert "whole name" in text
    assert "recognized" not in text


def test_overlay_null_check_does_not_ban_storage():
    patch = """\
diff --git a/util/OverlayMap.java b/util/OverlayMap.java
--- a/util/OverlayMap.java
+++ b/util/OverlayMap.java
 class OverlayMap {
- * Null values are not supported.
+ * Null values are not supported, neither in the base map nor in updates.
     public V get(Object key) {
-        if (additions.containsKey(key)) return additions.get(key);
+        V value = additions.get(key);
+        if (value != null) return value;
"""
    text = digest.line("", patch)
    assert "null check" in text
    assert "containsKey" in text
    assert "never stored" not in text


def test_load_compare_to_and_its_test():
    patch = """\
diff --git a/assignor/ProcessState.java b/assignor/ProcessState.java
--- a/assignor/ProcessState.java
+++ b/assignor/ProcessState.java
 class ProcessState {
-    public int compareTo(final ProcessState other) {
-        int loadCompare = Double.compare(this.load, other.load());
-        return Integer.compare(other.capacity, this.capacity);
-    }
diff --git a/src/test/java/assignor/ProcessStateTest.java b/src/test/java/assignor/ProcessStateTest.java
--- a/src/test/java/assignor/ProcessStateTest.java
+++ b/src/test/java/assignor/ProcessStateTest.java
-        assertTrue(processState1.compareTo(processState2) > 0);
"""
    text = digest.line("", patch)
    assert text.startswith("Remove ProcessState's load-based compareTo")
    assert "test" in text
    assert "unused" not in text


def test_big_endian_skips_non_llvm_variants():
    patch = """\
diff --git a/src/target.zig b/src/target.zig
--- a/src/target.zig
+++ b/src/target.zig
 fn selfHostedBackendIsAsRobustAsLlvm(target: *const std.Target) bool {
+    if (comptime builtin.cpu.arch.endian() == .big) return false; // https://github.com/ziglang/zig/issues/25961
diff --git a/test/src/ErrorTrace.zig b/test/src/ErrorTrace.zig
--- a/test/src/ErrorTrace.zig
+++ b/test/src/ErrorTrace.zig
 fn shouldTestNonLlvm(target: *const std.Target) bool {
+    if (comptime builtin.cpu.arch.endian() == .big) return false; // https://github.com/ziglang/zig/issues/25961
"""
    text = digest.line("", patch)
    assert "self-hosted backend" in text
    assert "non-LLVM" in text
    assert "25961" in text
    assert "stack trace" not in text.casefold()


def test_shared_base_search_names_hasBaseType():
    patch = """\
diff --git a/tsc/checker.go b/tsc/checker.go
--- a/tsc/checker.go
+++ b/tsc/checker.go
 func (c *Checker) hasBaseType(t *Type, checkBase *Type) bool {
+    // Searching each shared base type once keeps the search from going exponential.
+    return visited.AddIfAbsent(target) && core.Some(baseTypes, check)
-    return core.Some(c.getBaseTypes(target), check)
+Type 'Cycle1' recursively references itself as a base type.
"""
    text = digest.line("", patch)
    assert "hasBaseType" in text
    assert "recursive" in text
    assert "AddIfAbsent" not in text
    assert "memoize" not in text.casefold()


def test_formatter_javadoc_names_the_topic():
    patch = """\
diff --git a/tools/MirrorStateMessageFormatter.java b/tools/MirrorStateMessageFormatter.java
--- a/tools/MirrorStateMessageFormatter.java
+++ b/tools/MirrorStateMessageFormatter.java
+/**
+ * Formats records from the {@code __mirror_state} topic for the console consumer.
+ */
+public class MirrorStateMessageFormatter {
+    ((ObjectNode) json).put("state", MirrorPartitionState.nameOf(state.state()));
+    ((ObjectNode) json).put("previousState", MirrorPartitionState.nameOf(state.previousState()));
+}
"""
    text = digest.line("", patch)
    assert "MirrorStateMessageFormatter" in text
    assert "__mirror_state" in text
    assert "console consumer" in text
    assert "partition states" in text


def test_async_doc_says_called_not_started():
    patch = """\
diff --git a/lib/std/Io.zig b/lib/std/Io.zig
--- a/lib/std/Io.zig
+++ b/lib/std/Io.zig
+/// When this function returns, it is guaranteed that `function` has
+/// already been called and completed, or it has successfully been assigned
+/// a unit of concurrency.
 pub fn async(function: anytype) void {}
+/// When this function returns, it is guaranteed that `function` has already
+/// been called and completed, or it has successfully been assigned a unit of
+/// concurrency.
 pub fn groupAsync(function: anytype) void {}
"""
    text = digest.line("", patch)
    assert "called and completed" in text
    assert "unit of concurrency" in text
    assert "started" not in text.casefold()


def test_many_removed_wrappers_are_not_one_name():
    patch = """\
diff --git a/lib/std/os/windows.zig b/lib/std/os/windows.zig
--- a/lib/std/os/windows.zig
+++ b/lib/std/os/windows.zig
-pub fn GetFileAttributes(filename: []const u8) void {}
-pub fn FindClose(hFindFile: HANDLE) void {}
-pub fn CreateEventEx(flags: DWORD) void {}
-pub fn CreateEventExW(flags: DWORD) void {}
-pub fn VirtualAlloc(addr: ?*anyopaque) void {}
-pub fn VirtualFree(addr: ?*anyopaque) void {}
-pub fn GetFullPathNameW(path: [*:0]const u16) void {}
diff --git a/lib/std/os/windows/kernel32.zig b/lib/std/os/windows/kernel32.zig
--- a/lib/std/os/windows/kernel32.zig
+++ b/lib/std/os/windows/kernel32.zig
-pub extern "kernel32" fn FindClose(hFindFile: HANDLE) void;
-pub extern "kernel32" fn FindFirstFileW(name: LPCWSTR) HANDLE;
"""
    text = digest.line("", patch)
    assert "file, event, and virtual-memory" in text
    assert "windows.zig" in text
    assert "kernel32.zig" in text
    assert "ntdll" not in text.casefold()


def test_abort_resume_restore_is_removed():
    patch = """\
diff --git a/packages/react-dom/src/__tests__/ReactDOMFizzStaticBrowser-test.js b/packages/react-dom/src/__tests__/ReactDOMFizzStaticBrowser-test.js
--- a/packages/react-dom/src/__tests__/ReactDOMFizzStaticBrowser-test.js
+++ b/packages/react-dom/src/__tests__/ReactDOMFizzStaticBrowser-test.js
-  it('resumes a task that was aborted while rendering from where it started', async () => {
-    await ReactDOMFizzServer.resume(postponed);
-  });
diff --git a/packages/react-server/src/ReactFizzServer.js b/packages/react-server/src/ReactFizzServer.js
--- a/packages/react-server/src/ReactFizzServer.js
+++ b/packages/react-server/src/ReactFizzServer.js
-  const startKeyPath = task.keyPath;
-  const startChildIndex = task.childIndex;
-      task.keyPath = startKeyPath;
-      task.childIndex = startChildIndex;
"""
    text = digest.line("", patch)
    assert "keyPath" in text
    assert "childIndex" in text
    assert "restored" in text
    assert "restoring" not in text


def test_a_closed_channel_is_not_a_read_error():
    patch = """\
diff --git a/src/replication.c b/src/replication.c
--- a/src/replication.c
+++ b/src/replication.c
@@ void readReplBufferErrorHandler(connection *conn) {
-    serverLog(LL_WARNING, "Main channel error while reading from master: %s",
-              connGetLastError(conn));
+    if (connGetState(conn) == CONN_STATE_CLOSED) {
+        serverLog(LL_WARNING, "Main channel closed by master while reading");
+    } else {
+        serverLog(LL_WARNING, "Main channel error while reading from master: %s",
+                  connGetLastError(conn));
+    }
"""
    text = digest.line("", patch)
    assert "closed main channel" in text
    assert "read error" in text


def test_a_nan_comment_names_both_functions():
    import harness
    patch = """\
diff --git a/promql/durations.go b/promql/durations.go
--- a/promql/durations.go
+++ b/promql/durations.go
@@ func minOf() {
+// Match the min aggregator: NaN is only returned if both values are NaN.
+min_of(lhs, rhs)
+max_of(lhs, rhs)
"""
    files = harness._parse(patch)
    text = harness._widen(
        "Match the min aggregator: NaN is only returned if both values are NaN",
        files,
    )
    assert "min_of" in text and "max_of" in text
    assert "duration" in text.casefold()


def test_sort_keeps_listpack_instead_of_btree():
    patch = """\
diff --git a/src/sort.c b/src/sort.c
--- a/src/sort.c
+++ b/src/sort.c
@@ void sortCommandGeneric(client *c, int readonly) {
-    /* Destructively convert listpack encoded sorted sets for SORT. */
-    if (sortval->type == OBJ_ZSET && sortval->encoding == OBJ_ENCODING_LISTPACK) {
-        zsetConvert(sortval, OBJ_ENCODING_BTREE);
-    }
+    } else if (sortval->type == OBJ_ZSET && sortval->encoding == OBJ_ENCODING_LISTPACK) {
+        unsigned char *lp = sortval->ptr;
+    }
"""
    text = digest.line("", patch)
    assert "listpack" in text
    assert "btree" in text
    assert "SORT" in text
    assert "skiplist" not in text.casefold()


def test_merged_symbols_use_symbol_link_store():
    patch = """\
diff --git a/tsc/internal/checker/checker.go b/tsc/internal/checker/checker.go
--- a/tsc/internal/checker/checker.go
+++ b/tsc/internal/checker/checker.go
@@ type Checker struct {
-	mergedSymbols                               map[*ast.Symbol]*ast.Symbol
+	mergedSymbols                               symbolLinkStore[*ast.Symbol]
@@ func (c *Checker) cloneSymbol(symbol *ast.Symbol) *ast.Symbol {
-		merged := c.mergedSymbols[symbol]
+		if merged := c.mergedSymbols.TryGet(symbol); merged != nil && *merged != nil {
+			return *merged
"""
    text = digest.line("", patch)
    assert "symbolLinkStore" in text
    assert "TryGet" in text
    assert "concurrent" not in text.casefold()


def test_workspace_typescript_packages_are_not_a_global_default():
    patch = """\
diff --git a/src/tsdkPackage.ts b/src/tsdkPackage.ts
--- a/src/tsdkPackage.ts
+++ b/src/tsdkPackage.ts
-        const packagePath = joinPath(folder, "node_modules", "@typescript", "native-preview");
+export async function findWorkspaceTypeScriptVersions() {
+    const candidates = [
+        path.join(folder, "node_modules", "typescript"),
+        path.join(folder, "node_modules", "@typescript", "native"),
+        path.join(folder, "node_modules", "@typescript", "native-preview"),
+    ];
+}
"""
    text = digest.line("", patch)
    assert "typescript" in text
    assert "@typescript/native" in text
    assert "native-preview" in text
    assert "global" not in text.casefold()


def test_ip_quotas_go_through_the_socket_server():
    patch = """\
diff --git a/kafka/server/BrokerServer.scala b/kafka/server/BrokerServer.scala
--- a/kafka/server/BrokerServer.scala
+++ b/kafka/server/BrokerServer.scala
-      clientQuotaMetadataManager = new ClientQuotaMetadataManager(quotaManagers, socketServer.connectionQuotas)
+          (ip, quota) => socketServer.connectionQuotas.updateIpConnectionRateQuota(ip.toScala, quota.toScala),
diff --git a/metadata/ClientQuotaMetadataManager.scala b/metadata/ClientQuotaMetadataManager.scala
--- a/metadata/ClientQuotaMetadataManager.scala
+++ b/metadata/ClientQuotaMetadataManager.scala
-class ClientQuotaMetadataManager(private[metadata] val quotaManagers: QuotaManagers,
diff --git a/metadata/DynamicClientQuotaPublisher.java b/metadata/DynamicClientQuotaPublisher.java
--- a/metadata/DynamicClientQuotaPublisher.java
+++ b/metadata/DynamicClientQuotaPublisher.java
+public class DynamicClientQuotaPublisher implements MetadataPublisher {
"""
    text = digest.line("", patch)
    assert "DynamicClientQuotaPublisher" in text
    assert "ClientQuotaMetadataManager" in text
    assert "socketServer.connectionQuotas" in text
    assert "QuotaFactory" not in text


def test_unknown_user_is_compared_against_a_dummy_password():
    patch = """\
diff --git a/plain/PlainServerCallbackHandler.java b/plain/PlainServerCallbackHandler.java
--- a/plain/PlainServerCallbackHandler.java
+++ b/plain/PlainServerCallbackHandler.java
+    private static final char[] UNKNOWN_USER_DUMMY_PASSWORD = {'\\0'};
-            return expectedPassword != null && Utils.isEqualConstantTime(password, expectedPassword.toCharArray());
+            char[] comparedPassword = expectedPassword != null ? expectedPassword.toCharArray() : UNKNOWN_USER_DUMMY_PASSWORD;
+            boolean matches = Utils.isEqualConstantTime(password, comparedPassword);
+            return matches && expectedPassword != null;
"""
    text = digest.line("", patch)
    assert "isEqualConstantTime" in text
    assert "PlainServerCallbackHandler" in text
    assert "dummy password" in text
    assert "unknown" in text.casefold()


def test_get_string_returns_an_offset_error():
    patch = """\
diff --git a/decoder.go b/decoder.go
--- a/decoder.go
+++ b/decoder.go
-func (d *astDecoder) getString(idx uint32) string {
-	return d.allStringData[start:end]
+func (d *astDecoder) getString(idx uint32) (string, error) {
+	if offBase+8 > uint64(d.strData) {
+		return "", fmt.Errorf("invalid string index %d: offset pair exceeds string offsets section", idx)
+	}
+	if start > end || uint64(end) > uint64(len(d.allStringData)) {
+		return "", fmt.Errorf("invalid string offsets for index %d: [%d:%d] exceeds string data length %d", idx, start, end, len(d.allStringData))
+	}
"""
    text = digest.line("", patch)
    assert "getString" in text
    assert "string offsets section" in text
    assert "string data" in text
    assert "panic" not in text.casefold()


def test_rax_next_returns_instead_of_asserting():
    patch = """\
diff --git a/src/ebuckets.c b/src/ebuckets.c
--- a/src/ebuckets.c
+++ b/src/ebuckets.c
@@ int ebDefragRax(ebuckets *eb, EbucketsType *type, unsigned long *cursor,
-    assert(raxNext(&ri));
+    if (!raxNext(&ri)) {
+        *cursor = 0;
+        raxStop(&ri);
+        return 0;
+    }
+    TEST("item defragmentation - resume after the remaining buckets were removed") {
"""
    text = digest.line("", patch)
    assert "raxNext" in text
    assert "instead of asserting" in text
    assert "cursor" in text
    assert "resume" not in text.casefold()
    assert "crash" not in text.casefold()


def test_trusted_types_flag_stops_stringifying():
    patch = """\
diff --git a/shared/ReactFeatureFlags.js b/shared/ReactFeatureFlags.js
--- a/shared/ReactFeatureFlags.js
+++ b/shared/ReactFeatureFlags.js
-export const enableTrustedTypesIntegration: boolean = true;
diff --git a/client/ReactDOMComponent.js b/client/ReactDOMComponent.js
--- a/client/ReactDOMComponent.js
+++ b/client/ReactDOMComponent.js
-    enableTrustedTypesIntegration ? (value as any) : '' + (value as any),
+    value,
diff --git a/__tests__/ReactDOMForm-test.js b/__tests__/ReactDOMForm-test.js
--- a/__tests__/ReactDOMForm-test.js
+++ b/__tests__/ReactDOMForm-test.js
+    // Everything else is passed through without being stringified, so that Trusted Types values reach the DOM as-is.
"""
    text = digest.line("", patch)
    assert "enableTrustedTypesIntegration" in text
    assert "without stringifying" in text
    assert "Trusted Types" in text


def test_browser_export_drops_its_feature_flag():
    patch = """\
diff --git a/shared/ReactFeatureFlags.js b/shared/ReactFeatureFlags.js
--- a/shared/ReactFeatureFlags.js
+++ b/shared/ReactFeatureFlags.js
-export const enableBrowserAPI: boolean = true;
diff --git a/shared/ReactDOMBrowser.js b/shared/ReactDOMBrowser.js
--- a/shared/ReactDOMBrowser.js
+++ b/shared/ReactDOMBrowser.js
-const browserImpl = function browser(
-export const browser:
-  | void = enableBrowserAPI ? browserImpl : undefined;
+export function browser(reason?: ReactRecoverableReason): ReactRecoverable {
"""
    text = digest.line("", patch)
    assert "browser" in text
    assert "enableBrowserAPI" in text
    assert "unconditionally" in text


def test_flight_tasks_restore_the_previous_hooks_request():
    patch = """\
diff --git a/packages/react-server/src/ReactFlightServer.js b/packages/react-server/src/ReactFlightServer.js
--- a/packages/react-server/src/ReactFlightServer.js
+++ b/packages/react-server/src/ReactFlightServer.js
-  prepareToUseHooksForRequest(request);
-    resetHooksForRequest();
+  runWithRequest(request, retryTaskWithRequest, task);
+  runWithRequest(request, tryStreamTaskWithRequest, task);
+  runWithRequest(request, performWorkWithRequest, undefined);
diff --git a/packages/react-server/src/ReactFlightHooks.js b/packages/react-server/src/ReactFlightHooks.js
--- a/packages/react-server/src/ReactFlightHooks.js
+++ b/packages/react-server/src/ReactFlightHooks.js
-export function prepareToUseHooksForRequest(request: Request) {
+export function switchHooksRequest(request: null | Request): null | Request {
+  const previousRequest = currentRequest;
+  return previousRequest;
"""
    text = digest.line("", patch)
    assert "runWithRequest" in text
    assert "retryTask" in text
    assert "previous hooks request" in text
    assert "rejected" not in text.casefold()


def test_repeated_github_url_swap_names_the_deleted_snapshot():
    patch = """\
diff --git a/compiler/Cargo.toml b/compiler/Cargo.toml
--- a/compiler/Cargo.toml
+++ b/compiler/Cargo.toml
-repository = "https://github.com/facebook/react"
+repository = "https://github.com/react/react"
diff --git a/a.js b/a.js
--- a/a.js
+++ b/a.js
-href="https://github.com/facebook/react"
+href="https://github.com/react/react"
diff --git a/b.js b/b.js
--- a/b.js
+++ b/b.js
-href="https://github.com/facebook/react"
+href="https://github.com/react/react"
diff --git a/c.js b/c.js
--- a/c.js
+++ b/c.js
-href="https://github.com/facebook/react"
+href="https://github.com/react/react"
diff --git a/scripts/release/snapshot-test.js b/scripts/release/snapshot-test.js
--- a/scripts/release/snapshot-test.js
+++ b/scripts/release/snapshot-test.js
-console.log("old");
"""
    text = digest.line("", patch)
    assert "https://github.com/facebook/react" in text
    assert "https://github.com/react/react" in text
    assert "snapshot-test.js" in text
    assert "Reformat" not in text


def test_url_swap_scope_comes_from_the_stat():
    patch = """\
diff --git a/compiler/Cargo.toml b/compiler/Cargo.toml
--- a/compiler/Cargo.toml
+++ b/compiler/Cargo.toml
-repository = "https://github.com/facebook/react"
+repository = "https://github.com/react/react"
diff --git a/a.js b/a.js
--- a/a.js
+++ b/a.js
-href="https://github.com/facebook/react"
+href="https://github.com/react/react"
diff --git a/b.js b/b.js
--- a/b.js
+++ b/b.js
-href="https://github.com/facebook/react"
+href="https://github.com/react/react"
diff --git a/c.js b/c.js
--- a/c.js
+++ b/c.js
-href="https://github.com/facebook/react"
+href="https://github.com/react/react"
"""
    stat = """\
 compiler/Cargo.toml | 2 +-
 compiler/apps/playground/components/Header.tsx | 2 +-
 packages/react-devtools/README.md | 6 +-
 scripts/release/snapshot-test.js | 108 -
 scripts/release/snapshot-test.snapshot | 840 --------
"""
    text = digest.line(stat, patch)
    assert "playground" in text
    assert "devtools" in text
    assert "snapshot-test.js" in text
    assert "snapshot-test.snapshot" in text
    assert len(text) <= 170


def test_password_mask_covers_subclasses_and_parents():
    patch = """\
diff --git a/django/contrib/admin/utils.py b/django/contrib/admin/utils.py
--- a/django/contrib/admin/utils.py
+++ b/django/contrib/admin/utils.py
+        issubclass(field.model, AbstractBaseUser)
-    if field.name == "password" and field.model == get_user_model():
+    if field.name == "password" and issubclass(field.model, AbstractBaseUser):
diff --git a/docs/releases/6.0.txt b/docs/releases/6.0.txt
--- a/docs/releases/6.0.txt
+++ b/docs/releases/6.0.txt
+Password hash disclosure to "view only" admin users
"""
    text = digest.line("", patch)
    assert "AbstractBaseUser" in text
    assert "parents" in text
    assert "view only" in text


def test_ignore_warnings_message_is_literal_and_message_re_is_at_the_start():
    patch = """\
diff --git a/django/test/utils.py b/django/test/utils.py
--- a/django/test/utils.py
+++ b/django/test/utils.py
+def ignore_warnings(message=None, message_re=None):
+    Use the optional message argument to match a literal, case-sensitive
+    substring. Use message_re to match a case-insensitive regular expression at
+    the start of the warning message.
"""
    text = digest.line("", patch)
    assert "case-sensitive" in text
    assert "message_re" in text
    assert "at the start" in text


def test_multiline_tag_option_keeps_the_legacy_regex():
    patch = """\
diff --git a/django/template/base.py b/django/template/base.py
--- a/django/template/base.py
+++ b/django/template/base.py
+    def __init__(self, template_string, allow_multiline_tags=True):
+        self.tag_re = tag_re if allow_multiline_tags else tag_re_legacy
+        # RemovedInDjango2029Warning.
"""
    text = digest.line("", patch)
    assert "allow_multiline_tags" in text
    assert "tag_re_legacy" in text
    assert "RemovedInDjango2029Warning" in text


def test_block_tags_keep_raw_contents():
    patch = """\
diff --git a/django/template/base.py b/django/template/base.py
--- a/django/template/base.py
+++ b/django/template/base.py
-tag_re = re.compile(r"({%.*?%}|{{.*?}}|{#.*?#})")
+tag_re = re.compile(r"({%[\\s\\S]*?%}|{{.*?}}|{#.*?#})")
+        self.raw_contents = raw_contents
+        endpartialdef
"""
    text = digest.line("", patch)
    assert "raw_contents" in text
    assert "endpartialdef" in text
    assert "[\\s\\S]" in text


def test_geojson_crs_stays_ignored_on_oracle():
    patch = """\
diff --git a/django/contrib/gis/db/backends/mysql/features.py b/django/contrib/gis/db/backends/mysql/features.py
--- a/django/contrib/gis/db/backends/mysql/features.py
+++ b/django/contrib/gis/db/backends/mysql/features.py
+    def unsupported_geojson_options(self):
+        if self.connection.mysql_is_mariadb or self.connection.mysql_version < (9, 5):
+            return {"crs"}
diff --git a/docs/ref/contrib/gis/functions.txt b/docs/ref/contrib/gis/functions.txt
--- a/docs/ref/contrib/gis/functions.txt
+++ b/docs/ref/contrib/gis/functions.txt
+                       GeoJSON. Ignored on MySQL < 9.5, MariaDB, and Oracle.
"""
    text = digest.line("", patch)
    assert "GeoJSON" in text
    assert "Oracle" in text
    assert "MariaDB" in text


def test_listpack_hash_field_stays_in_place():
    patch = """\
diff --git a/src/t_hash.c b/src/t_hash.c
--- a/src/t_hash.c
+++ b/src/t_hash.c
+    listpackEx *lpt = o->ptr;
+    int inplace;
+    A persistent field sorts after every volatile field.
+    if (expire_at == HASH_LP_NO_TTL)
"""
    text = digest.line("", patch)
    assert "in place" in text
    assert "listpack" in text
    assert "delete and re-insert" in text


def test_auth_disconnect_releases_private_data():
    patch = """\
diff --git a/src/module.c b/src/module.c
--- a/src/module.c
+++ b/src/module.c
+void moduleFreeAuthBlockedClient(client *c) {
+    /* must still release private data and the module reference. */
"""
    text = digest.line("", patch)
    assert "private data" in text
    assert "module reference" in text
    assert "AUTH" in text


def test_loongarch_clock_needs_llftp():
    patch = """\
diff --git a/src/monotonic.c b/src/monotonic.c
--- a/src/monotonic.c
+++ b/src/monotonic.c
+#if defined(__loongarch_lp64) && defined(__linux__)
+    __asm__ __volatile__("rdtime.d %0, $r0": "=r" (val));
+#define CPUCFG2_LLFTP       (1 << 14)
"""
    text = digest.line("", patch)
    assert "rdtime.d" in text
    assert "LLFTP" in text
    assert "Linux" in text


def test_parallel_transitions_keep_the_entangled_lanes():
    patch = """\
diff --git a/packages/react-reconciler/src/ReactFiberLane.js b/packages/react-reconciler/src/ReactFiberLane.js
--- a/packages/react-reconciler/src/ReactFiberLane.js
+++ b/packages/react-reconciler/src/ReactFiberLane.js
-      if (enableParallelTransitions) {
-        return getHighestPriorityLane(lanes);
-      }
-      return lanes & TransitionUpdateLanes;
+      return getHighestPriorityLane(lanes);
+  suspendedLanes = getEntangledLanes(root, suspendedLanes);
"""
    text = digest.line("", patch)
    assert "enableParallelTransitions" in text
    assert "entangled" in text
    assert "highest-priority" in text


def test_effect_event_leaves_the_before_mutation_phase():
    patch = """\
diff --git a/packages/react-reconciler/src/ReactFiberCommitWork.js b/packages/react-reconciler/src/ReactFiberCommitWork.js
--- a/packages/react-reconciler/src/ReactFiberCommitWork.js
+++ b/packages/react-reconciler/src/ReactFiberCommitWork.js
-export const enableEffectEventMutationPhase = true;
-function commitBeforeMutationEffectsOnFiber() {
-      if (!enableEffectEventMutationPhase) {
-        ref.impl = nextImpl;
-      }
+        ref.impl = nextImpl;
"""
    text = digest.line("", patch)
    assert "before-mutation" in text
    assert "ref.impl" in text
    assert "enableEffectEventMutationPhase" in text


def test_fragment_siblings_include_host_singleton():
    patch = """\
diff --git a/packages/shared/ReactFeatureFlags.js b/packages/shared/ReactFeatureFlags.js
--- a/packages/shared/ReactFeatureFlags.js
+++ b/packages/shared/ReactFeatureFlags.js
-export const enableFragmentRefsTextNodes = true;
-export const enableFragmentRefsInstanceHandles = true;
diff --git a/packages/react-reconciler/src/ReactFiberTreeReflection.js b/packages/react-reconciler/src/ReactFiberTreeReflection.js
--- a/packages/react-reconciler/src/ReactFiberTreeReflection.js
+++ b/packages/react-reconciler/src/ReactFiberTreeReflection.js
+ * Collects HostComponent, HostSingleton, and HostText siblings.
"""
    text = digest.line("", patch)
    assert "HostSingleton" in text
    assert "enableFragmentRefsTextNodes" in text
    assert "enableFragmentRefsInstanceHandles" in text


def test_prefixed_jemalloc_leaves_libc_thread_hooks():
    patch = """\
diff --git a/deps/jemalloc/configure.ac b/deps/jemalloc/configure.ac
--- a/deps/jemalloc/configure.ac
+++ b/deps/jemalloc/configure.ac
+dnl A prefixed jemalloc lives alongside the libc malloc, and defining these
+dnl hooks would interpose the libc malloc's own, so that it never cleans up
+dnl exited threads.
+if test "x${JEMALLOC_PREFIX}" = "x" ; then
"""
    text = digest.line("", patch)
    assert "prefixed" in text
    assert "exited threads" in text
    assert "libc malloc" in text


def test_dynamic_import_option_is_for_non_literal_calls():
    patch = """\
diff --git a/tsc/internal/tsoptions/options_generated.go b/tsc/internal/tsoptions/options_generated.go
--- a/tsc/internal/tsoptions/options_generated.go
+++ b/tsc/internal/tsoptions/options_generated.go
+    Name: "disableDynamicImportExtensionRewriting",
+    Description: "Disable runtime rewriting of file extensions in non-literal import() and require() calls when rewriteRelativeImportExtensions is enabled.",
"""
    text = digest.line("", patch)
    assert "non-literal" in text
    assert "disableDynamicImportExtensionRewriting" in text
    assert "rewriteRelativeImportExtensions" in text
    assert len(text) <= 170


def test_memory_ratio_counts_replication_buffers():
    patch = """\
diff --git a/src/module.c b/src/module.c
--- a/src/module.c
+++ b/src/module.c
+ * replication and AOF buffers that RedisModule_GetUsedMemoryRatio() excludes
+ * are counted. It is thread safe.
+float RM_GetUsedMemoryRatioEstimate(void) {
+    atomicGet(maxmemory_snapshot, maxmemory);
"""
    text = digest.line("", patch)
    assert "RM_GetUsedMemoryRatioEstimate" in text
    assert "AOF buffers" in text
    assert "thread-safe" in text


def test_aof_rewrite_backs_off_after_a_failed_flush():
    patch = """\
diff --git a/src/aof.c b/src/aof.c
--- a/src/aof.c
+++ b/src/aof.c
+            "Can't flush the AOF buffer to disk. Background AOF rewrite aborted.");
+        /* Count it as a failure so scheduled rewrites back off via aofRewriteLimited(). */
+    /* DEBUG AOF-FLUSH-FORCE fault injection. */
"""
    text = digest.line("", patch)
    assert "AOF-FLUSH-FORCE" in text
    assert "count that failure" in text
    assert "cannot be flushed" in text
    assert "back off" not in text


def test_lua_matches_can_be_killed():
    patch = """\
diff --git a/deps/lua/src/lstrlib.c b/deps/lua/src/lstrlib.c
--- a/deps/lua/src/lstrlib.c
+++ b/deps/lua/src/lstrlib.c
+ * pattern matcher, so long matches stay interruptible by SCRIPT KILL / FUNCTION KILL.
"""
    text = digest.line("", patch)
    assert "FUNCTION KILL" in text
    assert "SCRIPT KILL" in text
    assert "interruptible" in text


def test_asm_clients_keep_their_own_output_buffers():
    patch = """\
diff --git a/src/networking.c b/src/networking.c
--- a/src/networking.c
+++ b/src/networking.c
+     * set below repl-backlog-size. ASM clients use the replica limits but have their own
+     * output buffers, so their limits are not adjusted. */
+    if (clientTypeIsSlave(c) && hard_limit_bytes &&
+        hard_limit_bytes < (unsigned long long)server.repl_backlog_size)
+        hard_limit_bytes = server.repl_backlog_size;
+    unsigned long long soft_limit_bytes = server.client_obuf_limits[class].soft_limit_bytes;
"""
    text = digest.line("", patch)
    assert "ASM" in text
    assert "repl-backlog-size" in text
    assert "soft and hard" in text
    assert len(text) <= 170


def test_minid_trim_propagates_the_first_live_id():
    patch = """\
diff --git a/src/t_stream.c b/src/t_stream.c
--- a/src/t_stream.c
+++ b/src/t_stream.c
+ * For MINID we propagate the successor of 'max_deleted_id', the largest ID the
+ * trim removed. When nothing was deleted (possible through XADD) it is NULL and
+ * we propagate the first live ID, so replaying the exact trim is a no-op.
"""
    text = digest.line("", patch)
    assert "first live ID" in text
    assert "XADD" in text
    assert "MINID" in text
    assert len(text) <= 170


def test_arlastitems_caps_nulls_on_a_sparse_array():
    patch = """\
diff --git a/src/t_array.c b/src/t_array.c
--- a/src/t_array.c
+++ b/src/t_array.c
+void arlastitemsCommand(client *c) {
+    /* We walk positions, not existing items, so on a sparse array the reply
+     * can be made of a huge amount of NULLs. We cap it. */
"""
    text = digest.line("", patch)
    assert "ARLASTITEMS" in text
    assert "NULLs" in text
    assert "sparse" in text


def test_zrange_stops_updating_alloc_size_in_store_mode():
    patch = """\
diff --git a/src/t_zset.c b/src/t_zset.c
--- a/src/t_zset.c
+++ b/src/t_zset.c
-        updateSlotAllocSize(c->db, getKeySlot(k
+     * in store mode it may be the destination, which setKey()/dbDelete() replace/free. */
"""
    text = digest.line("", patch)
    assert "setKey" in text
    assert "dbDelete" in text
    assert "ZRANGE" in text


def test_cluster_bus_checks_the_full_extension_length():
    patch = """\
diff --git a/src/cluster_legacy.c b/src/cluster_legacy.c
--- a/src/cluster_legacy.c
+++ b/src/cluster_legacy.c
+        # its low 16 bits (8) would pass a truncated check; the full length must not.
+        # gossip section and extension length
"""
    text = digest.line("", patch)
    assert "16-bit" in text
    assert "gossip" in text
    assert "extension" in text
    assert len(text) <= 170


def test_xtrim_acked_keeps_scanning_referenced_entries():
    patch = """\
diff --git a/src/t_stream.c b/src/t_stream.c
--- a/src/t_stream.c
+++ b/src/t_stream.c
+         * or some entries were skipped because they are still referenced), continue
+         * if we didn't reach the trim limit within this node
+        r XTRIM mystream MAXLEN = 3 ACKED
"""
    text = digest.line("", patch)
    assert "still referenced" in text
    assert "XTRIM" in text
    assert "trim limit" in text
    assert len(text) <= 170


def test_localization_update_keeps_the_checkout():
    patch = """\
diff --git a/tools/pipelines/localization-update.yml b/tools/pipelines/localization-update.yml
--- a/tools/pipelines/localization-update.yml
+++ b/tools/pipelines/localization-update.yml
+        fetchDepth: 0
diff --git a/tools/scripts/preserve-localization-base.mjs b/tools/scripts/preserve-localization-base.mjs
--- a/tools/scripts/preserve-localization-base.mjs
+++ b/tools/scripts/preserve-localization-base.mjs
+    const base = git("merge-base", main, previousHead);
+    // Apply only the generated delta to the old base, without changing the checkout.
"""
    text = digest.line("", patch)
    assert "checkout" in text
    assert "merge-base" in text
    assert "generated delta" in text
    assert len(text) <= 170


def test_signed_zero_stays_distinct_for_babel():
    patch = """\
diff --git a/compiler/crates/react_compiler_optimization/src/constant_propagation.rs b/compiler/crates/react_compiler_optimization/src/constant_propagation.rs
--- a/compiler/crates/react_compiler_optimization/src/constant_propagation.rs
+++ b/compiler/crates/react_compiler_optimization/src/constant_propagation.rs
@@ -252,6 +252,13 @@ fn evaluate_phi(phi: &Phi, constants: &Constants) -> Option<Constant> {
+                    // Strict equality alone would merge 0 and -0, which are not
+                    // interchangeable.
diff --git a/compiler/crates/react_codegen/src/lib.rs b/compiler/crates/react_codegen/src/lib.rs
--- a/compiler/crates/react_codegen/src/lib.rs
+++ b/compiler/crates/react_codegen/src/lib.rs
+                // Emit negative zero explicitly so Babel preserves its sign.
"""
    text = digest.line("", patch)
    assert "phi" in text
    assert "Babel" in text
    assert "-0" in text
    assert len(text) <= 170


def test_qpc_ticks_use_fixed_point_past_the_common_frequency():
    patch = """\
diff --git a/lib/std/time.zig b/lib/std/time.zig
--- a/lib/std/time.zig
+++ b/lib/std/time.zig
+            const qpc = windows.QueryPerformanceCounter();
+            const common_qpf = 10_000_000;
+            // Convert to ns using fixed point.
"""
    text = digest.line("", patch)
    assert "QPC" in text
    assert "10_000_000" in text
    assert "fixed point" in text
    assert len(text) <= 170


def test_aborted_transactions_scan_from_the_start_segment():
    patch = """\
diff --git a/storage/src/main/java/org/apache/kafka/storage/internals/log/UnifiedLog.java b/UnifiedLog.java
--- a/storage/src/main/java/org/apache/kafka/storage/internals/log/UnifiedLog.java
+++ b/storage/src/main/java/org/apache/kafka/storage/internals/log/UnifiedLog.java
+     * Collect the aborted transactions that may have data in [startOffset, upperBoundOffset): those ending at or after
+     * startOffset and starting before upperBoundOffset, scanning the transaction indexes from the segment containing
+     * startOffset.
+    public List<AbortedTxn> collectAbortedTransactions(long startOffset, long upperBoundOffset) {
"""
    text = digest.line("", patch)
    assert "startOffset" in text
    assert "upperBoundOffset" in text
    assert "segment that contains" in text
    assert len(text) <= 170


def test_zadd_shrinks_the_member_index_after_skipped_pairs():
    patch = """\
diff --git a/src/t_zset.c b/src/t_zset.c
--- a/src/t_zset.c
+++ b/src/t_zset.c
+ * ZADD's hint counts its score-member pairs, and the B+ tree member index is
+ * presized for them. The pairs may repeat a member or be skipped by
+ * NX/XX/GT/LT, so the caller must call zbtreeShrinkToFit() once they are
"""
    text = digest.line("", patch)
    assert "presized" in text
    assert "NX, XX, GT, or LT" in text
    assert "ZADD" in text
    assert len(text) <= 170


def test_dict_link_is_refreshed_after_a_rehash():
    patch = """\
diff --git a/src/db.c b/src/db.c
--- a/src/db.c
+++ b/src/db.c
+ * update `*plink`. Use before reusing a link after code that
+ * may look up keys (KSN, module hooks): a lookup may run a rehash step that
+ * moves entries. `kv` may already be freed.
+void dbRefreshLink(void) {
"""
    text = digest.line("", patch)
    assert "dbRefreshLink" in text
    assert "rehash" in text
    assert "freed" in text
    assert len(text) <= 170


def test_fpconv_const_tables_become_multiply_shifts():
    patch = """\
diff --git a/deps/fpconv/Makefile b/deps/fpconv/Makefile
--- a/deps/fpconv/Makefile
+++ b/deps/fpconv/Makefile
+OPT= -O3
+/* Must stay const: generate_digits() divides part1 by tens[i], and the compiler
+ * can only replace those divisions with multiply-shifts if it may assume the
+ * table's contents. */
+static const uint64_t tens[] = { 10U, 1U };
"""
    text = digest.line("", patch)
    assert "-O3" in text
    assert "multiply-shifts" in text
    assert "fpconv" in text
    assert len(text) <= 170


def test_workgroup_limit_runs_on_the_caller():
    patch = """\
diff --git a/core/workgroup.go b/core/workgroup.go
--- a/core/workgroup.go
+++ b/core/workgroup.go
+wg := core.NewWorkGroup(core.WorkGroupConcurrency(singleThreaded))
+// A concurrency of 1 runs the functions on the goroutine that calls RunAndWait; 0 means no limit.
+// Running functions call Queue, so waiting for a slot there would deadlock at the limit.
"""
    text = digest.line("", patch)
    assert "RunAndWait" in text
    assert "0 means no limit" in text
    assert "deadlock" in text
    assert "all parallel" not in text.casefold()
    assert len(text) <= 170


def test_literal_fffd_is_not_a_one_byte_rune_error():
    patch = """\
diff --git a/regexp.go b/regexp.go
--- a/regexp.go
+++ b/regexp.go
+// A literal U+FFFD in a regex character-class range must not be mistaken for
+// utf8.RuneError (which is the same code point, but only signals invalid UTF-8
+// when DecodeRune returns size 1).
+if ch == utf8.RuneError && size == 1 {
"""
    text = digest.line("", patch)
    assert "U+FFFD" in text
    assert "character-class" in text
    assert "size 1" in text
    assert len(text) <= 170


def test_offset_reset_factories_are_both_removed():
    patch = """\
diff --git a/group/ShareGroupAutoOffsetResetStrategy.java b/group/ShareGroupAutoOffsetResetStrategy.java
--- a/group/ShareGroupAutoOffsetResetStrategy.java
+++ b/group/ShareGroupAutoOffsetResetStrategy.java
-    public static ShareGroupAutoOffsetResetStrategy earliest() {
-        return new ShareGroupAutoOffsetResetStrategy(AutoOffsetResetStrategy.EARLIEST, StrategyType.EARLIEST);
-    }
-    public static ShareGroupAutoOffsetResetStrategy latest() {
-        return new ShareGroupAutoOffsetResetStrategy(AutoOffsetResetStrategy.LATEST, StrategyType.LATEST);
-    }
"""
    text = digest.line("", patch)
    assert "earliest()" in text
    assert "latest()" in text
    assert "unused" not in text.casefold()
    assert len(text) <= 170


def test_mlkem_vec_is_renamed_to_polyvec():
    patch = """\
diff --git a/lib/std/crypto/ml_kem.zig b/lib/std/crypto/ml_kem.zig
--- a/lib/std/crypto/ml_kem.zig
+++ b/lib/std/crypto/ml_kem.zig
-        const V = Vec(p.k);
+        const V = PolyVec(p.k);
-            pub const bytes_length = InnerPk.bytes_length;
+            pub const encoded_length = InnerPk.encoded_length;
"""
    text = digest.line("", patch)
    assert "PolyVec" in text
    assert "encoded_length" in text
    assert "Euclid" not in text
    assert len(text) <= 170


def test_process_state_load_becomes_private():
    patch = """\
diff --git a/streams/ProcessState.java b/streams/ProcessState.java
--- a/streams/ProcessState.java
+++ b/streams/ProcessState.java
-    public void computeLoad() {
+    private void computeLoad() {
-    public Set<TaskId> assignedActiveTasks() {
-    public boolean hasCapacity() {
"""
    text = digest.line("", patch)
    assert "computeLoad" in text
    assert "private" in text
    assert "unused" not in text.casefold()
    assert len(text) <= 170


def test_effect_event_fixture_names_the_issue():
    patch = """\
diff --git a/fixtures/useEffectEvent-capture-variable-declared-later.js b/fixtures/useEffectEvent.js
--- a/fixtures/useEffectEvent-capture-variable-declared-later.js
+++ b/fixtures/useEffectEvent-capture-variable-declared-later.js
+// Repro from https://github.com/facebook/react/issues/37209
+// The effect event reads `length`, which is declared after the callback.
+const logLength = useEffectEvent(() => {
"""
    text = digest.line("", patch)
    assert "37209" in text
    assert "useEffectEvent" in text
    assert "declared after" in text
    assert len(text) <= 170


def test_zrangestore_rev_appends_into_a_btree():
    patch = """\
diff --git a/src/t_zset.c b/src/t_zset.c
--- a/src/t_zset.c
+++ b/src/t_zset.c
+     * ZRANGESTORE walks it forward when the destination is a B+ tree.
+     * A listpack destination has no append path. */
"""
    text = digest.line("", patch)
    assert "ZRANGESTORE" in text
    assert "listpack destination" in text
    assert "REV" in text


def test_thread_pool_becomes_io_group():
    patch = """\
diff --git a/lib/compiler/build_runner.zig b/lib/compiler/build_runner.zig
--- a/lib/compiler/build_runner.zig
+++ b/lib/compiler/build_runner.zig
-    var thread_pool_options: std.Thread.Pool.Options = .{};
+                threaded.setAsyncLimit(.limited(n));
+        var group: Io.Group = .init;
+        fuzz.group.concurrent(io, coverageRun, .{fuzz});
"""
    text = digest.line("", patch)
    assert "std.Thread.Pool" in text
    assert "setAsyncLimit" in text
    assert "group.concurrent" in text

