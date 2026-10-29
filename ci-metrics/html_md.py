"""Turn the rendered report HTML into Markdown.

report.py builds the report as HTML strings, so the Markdown is produced from
that same output instead of from a second renderer: there is one set of numbers
and one set of prose, and a new section shows up in both files for free.

Only the tag vocabulary report.py actually emits is handled (headings,
paragraphs, tables, and the inline b/i/em/code/a/span). Anything else degrades
to its text. Cell classes carry meaning in the HTML - `win` marks a ROCm
like-for-like win, `dominant` the largest phase of a row, `pending` a figure
that was not measured - so they become bold, bold and italic here.
"""
import re
from html.parser import HTMLParser

SKIP = {"head", "style", "title", "script"}
HEADING = {"h1": "#", "h2": "##", "h3": "###"}
WRAP = {"b": "**", "strong": "**", "i": "*", "em": "*", "code": "`"}
WIN = "▲"


def _cell(text, cls):
    """Re-add in Markdown what the HTML said with a class."""
    t = re.sub(r"\s+", " ", text).strip() or " "
    if "dominant" in cls or "win" in cls:
        t = f"**{t}**"
    if "win" in cls:
        t += " " + WIN
    if "pending" in cls:
        t = f"*{t}*"
    return t.replace("|", "\\|")


class _MD(HTMLParser):
    def __init__(self):
        # convert_charrefs resolves &nbsp; / &middot; for us, so nothing below
        # unescapes a second time.
        super().__init__(convert_charrefs=True)
        self.out = []        # finished Markdown blocks
        self.buf = []        # text of the block being built
        self.skip = 0
        self.head = None     # prefix for the block being built ("##", ">")
        self.table = None    # (header_rows, body_rows) while inside a table
        self.row = None
        self.cell = None
        self.cell_cls = ""
        self.cell_span = 1
        self.carry = []      # [col, text, rows_left] for each open rowspan
        self.new_spans = []
        self.spans = []      # one entry per open <span>: True if it is a .scope tag
        self.chip = 0        # inside the colour key's chip, which has no Markdown equivalent
        self.href = ""

    # -- helpers --
    def _flush(self):
        t = re.sub(r"[ \t ]+", " ", "".join(self.buf)).strip()
        self.buf = []
        head, self.head = self.head, None
        if t:
            self.out.append(f"{head} {t}" if head else t)

    def _text(self, s):
        (self.cell if self.cell is not None else self.buf).append(s)

    # -- parser hooks --
    def handle_starttag(self, tag, attrs):
        if tag in SKIP:
            self.skip += 1
            return
        if self.skip:
            return
        a = dict(attrs)
        cls = a.get("class", "")
        if tag in HEADING:
            self._flush()
            self.head = HEADING[tag]
        elif tag == "p":
            self._flush()
            # .note is an aside in the HTML; keep it set apart.
            self.head = ">" if "note" in cls else None
        elif tag == "div":
            # div.defs / div.key are blocks of their own in the HTML layout.
            self._flush()
        elif tag == "table":
            self._flush()
            self.table = ([], [])
        elif tag == "tr":
            self.row = []
            self.new_spans = []
        elif tag in ("td", "th"):
            self.cell = []
            self.cell_cls = cls
            self.cell_span = int(a.get("rowspan", 1))
        elif tag == "a":
            self.href = a.get("href", "")
            self._text("[")
        elif tag == "span":
            scope = "scope" in cls
            self.spans.append(scope)
            if "chip" in cls:
                # The key says "green"; in Markdown the win marker is the arrow.
                self.chip += 1
                self._text(f"**{WIN}**")
            elif scope:
                self._text(" (")
            elif "".join(self.cell if self.cell is not None else self.buf).strip():
                # Siblings of a flex row (the colour key) would otherwise run
                # together once the layout is gone.
                self._text(" · ")
        elif tag == "br":
            self._text(" ")
        elif tag in WRAP:
            self._text(WRAP[tag])

    def handle_endtag(self, tag):
        if tag in SKIP:
            self.skip = max(0, self.skip - 1)
            return
        if self.skip:
            return
        if tag in HEADING or tag in ("p", "div"):
            self._flush()
        elif tag in ("td", "th"):
            if self.row is not None:
                if self.cell_span > 1:
                    self.new_spans.append([len(self.row), "".join(self.cell), self.cell_span - 1])
                self.row.append(_cell("".join(self.cell), self.cell_cls))
            self.cell = None
            self.cell_span = 1
        elif tag == "tr" and self.table is not None:
            # Markdown has no rowspan, so repeat the spanned value in every row
            # it covers; that keeps the column count right.
            for c in self.carry:
                c[2] -= 1
                self.row.insert(min(c[0], len(self.row)), _cell(c[1], ""))
            self.carry = [c for c in self.carry if c[2] > 0] + self.new_spans
            self.table[1 if self.table[0] else 0].append(self.row)
            self.row = None
        elif tag == "table":
            self.out.append(_render_table(*self.table))
            self.table, self.carry = None, []
        elif tag == "span":
            if self.spans.pop() if self.spans else False:
                self._text(")")
            self.chip = max(0, self.chip - 1)
        elif tag == "a":
            self._text(f"]({self.href})")
        elif tag in WRAP:
            self._text(WRAP[tag])

    def handle_data(self, d):
        if not self.skip and not self.chip:
            self._text(d)


def _render_table(header, body):
    if not header:
        return ""
    head = header[0]
    n = len(head)
    rows = [r + [" "] * (n - len(r)) if len(r) < n else r[:n] for r in body]

    def line(cells):
        return "| " + " | ".join(cells) + " |"

    return "\n".join([line(head), line(["---"] * n)] + [line(r) for r in rows])


def to_markdown(doc):
    """Render report.py's HTML document as Markdown."""
    p = _MD()
    p.feed(doc)
    p.close()
    p._flush()
    blocks = [re.sub(r"[ \t ]+", " ", b).strip() for b in p.out]
    return "\n\n".join(b for b in blocks if b) + "\n"
