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


def test_pick_keeps_a_faithful_second_clause():
    import harness
    draft = "Restore tab stops if we have tab optimizations enabled"
    better = "Restore tab stops if hard tabs are enabled; otherwise, set them to -1"
    assert harness.pick(better, draft, f"draft: {draft}") == better
