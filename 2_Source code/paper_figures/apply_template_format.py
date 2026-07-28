"""Post-process the pandoc-generated DOCX so it matches the IEEE Access template
exactly: two-column body, PARA body style, bold-paragraph headings, AU title,
DOP header lines, and column-width figures."""
import re, shutil, zipfile, os, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "Trustworthy_Machine_Unlearning_Paper.docx"
DST = sys.argv[2] if len(sys.argv) > 2 else "Trustworthy_Machine_Unlearning_Paper_formatted.docx"
WORK = "/tmp/_fmt"

# ---- section properties copied from the template -------------------------
SECT1 = ('<w:sectPr><w:pgSz w:w="11520" w:h="15660" w:code="1"/>'
         '<w:pgMar w:top="1280" w:right="740" w:bottom="1040" w:left="740" '
         'w:header="360" w:footer="500" w:gutter="0"/>'
         '<w:cols w:space="720"/><w:docGrid w:linePitch="360"/></w:sectPr>')
SECT2 = ('<w:sectPr><w:type w:val="continuous"/>'
         '<w:pgSz w:w="11520" w:h="15660" w:code="1"/>'
         '<w:pgMar w:top="1300" w:right="740" w:bottom="1040" w:left="740" '
         'w:header="360" w:footer="640" w:gutter="0"/>'
         '<w:cols w:num="2" w:space="400"/><w:docGrid w:linePitch="360"/></w:sectPr>')

HEAD_PPR = ('<w:pPr><w:spacing w:before="260" w:after="160" w:line="278" w:lineRule="auto"/>'
            '<w:rPr><w:b/><w:bCs/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr></w:pPr>')
HEAD_RPR = '<w:b/><w:bCs/><w:sz w:val="20"/><w:szCs w:val="20"/>'
SUB_PPR = ('<w:pPr><w:pStyle w:val="PARA"/>'
           '<w:spacing w:before="200" w:after="80" w:line="240" w:lineRule="auto"/>'
           '<w:rPr><w:b/><w:bCs/></w:rPr></w:pPr>')
SUB_RPR = '<w:b/><w:bCs/>'

COL_EMU_W = 3000000                     # ~3.28 in, fits one column
shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK)
with zipfile.ZipFile(SRC) as z:
    z.extractall(WORK)
p = os.path.join(WORK, "word", "document.xml")
doc = open(p, encoding="utf-8").read()


def ptext(par):
    return "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", par))


def force_run_props(par, props):
    """Ensure every run in the paragraph carries the given run properties."""
    def fix(m):
        run = m.group(0)
        if "<w:rPr>" in run:
            return run.replace("<w:rPr>", "<w:rPr>" + props, 1)
        open_tag = re.match(r"<w:r\b[^>]*>", run).group(0)
        return run.replace(open_tag, open_tag + "<w:rPr>" + props + "</w:rPr>", 1)
    return re.sub(r"<w:r\b[^>]*>.*?</w:r>", fix, par, flags=re.S)


def set_ppr(par, ppr):
    if re.search(r"<w:pPr>.*?</w:pPr>", par, re.S):
        return re.sub(r"<w:pPr>.*?</w:pPr>", ppr, par, count=1, flags=re.S)
    return re.sub(r"(<w:p\b[^>]*>)", r"\1" + ppr, par, count=1)


PARA_RE = r"<w:p\b[^>]*/>|<w:p\b[^>]*>.*?</w:p>"
tbl_spans = [(m.start(), m.end()) for m in re.finditer(r"<w:tbl>.*?</w:tbl>", doc, re.S)]
def in_table(pos):
    return any(a <= pos < b for a, b in tbl_spans)
matches = list(re.finditer(PARA_RE, doc, re.S))
paras = [m.group(0) for m in matches]
intbl = [in_table(m.start()) for m in matches]
out = []
seen_index_terms = False
for idx, par in enumerate(paras):
    t = ptext(par).strip()
    is_tbl = intbl[idx]
    style = re.search(r'<w:pStyle w:val="([^"]+)"\s*/>', par)
    style = style.group(1) if style else ""

    # 1) header lines -> DOP
    if t.startswith(("Date of publication", "Digital Object Identifier")):
        par = set_ppr(par, '<w:pPr><w:pStyle w:val="DOP"/></w:pPr>')

    # 2) title and author line -> AU (template colour and size)
    elif style == "Heading1" or t.startswith("Author One"):
        par = set_ppr(par, '<w:pPr><w:pStyle w:val="AU"/><w:spacing w:after="0"/></w:pPr>')
        par = force_run_props(par, '<w:color w:val="00629B"/><w:sz w:val="44"/><w:szCs w:val="44"/>')

    # 3) INDEX TERMS closes the one-column section
    elif t.startswith("INDEX TERMS"):
        par = set_ppr(par, '<w:pPr><w:pStyle w:val="IT"/>' + SECT1 + "</w:pPr>")
        seen_index_terms = True

    # 4) ABSTRACT keeps the template's spacing at 10pt
    elif t.startswith("ABSTRACT"):
        par = set_ppr(par, '<w:pPr><w:spacing w:after="160" w:line="278" w:lineRule="auto"/>'
                           '<w:rPr><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr></w:pPr>')
        par = force_run_props(par, '<w:sz w:val="20"/><w:szCs w:val="20"/>')

    # 5) numbered section headings -> bold 10pt paragraph
    elif style == "Heading2":
        par = set_ppr(par, HEAD_PPR)
        par = force_run_props(par, HEAD_RPR)

    # 6) lettered subsection headings -> PARA + bold
    elif style in ("Heading3", "Heading4", "Heading5"):
        par = set_ppr(par, SUB_PPR)
        par = force_run_props(par, SUB_RPR)

    # 7) ordinary body text -> PARA.
    # NOTE: the template's PARA style pins line spacing to "exact" 240 twips, which
    # clips anything taller than a line of text, so inline images end up sitting
    # under the following paragraph. We override the line rule to "auto" and add a
    # little space after each paragraph so the column is readable rather than solid.
    elif style in ("BodyText", "FirstParagraph", "Compact", "SourceCode") or style == "":
        if "<w:drawing>" in par:                       # figure
            par = set_ppr(par, '<w:pPr><w:pStyle w:val="PARA"/><w:jc w:val="center"/>'
                               '<w:spacing w:before="180" w:after="60" w:line="240" '
                               'w:lineRule="auto"/></w:pPr>')
        elif t.startswith("Figure ") or t.startswith("Table "):   # caption
            par = set_ppr(par, '<w:pPr><w:pStyle w:val="PARA"/><w:jc w:val="center"/>'
                               '<w:spacing w:before="0" w:after="200" w:line="240" '
                               'w:lineRule="auto"/></w:pPr>')
        elif t and is_tbl:                             # table cell
            par = set_ppr(par, '<w:pPr><w:pStyle w:val="PARA"/>'
                               '<w:spacing w:after="0" w:line="240" '
                               'w:lineRule="auto"/></w:pPr>')
        elif t:                                        # body paragraph
            par = set_ppr(par, '<w:pPr><w:pStyle w:val="PARA"/>'
                               '<w:spacing w:after="120" w:line="240" '
                               'w:lineRule="auto"/></w:pPr>')
    out.append(par)

# rebuild the body positionally (replace-by-value is unsafe: identical paragraphs collide)
_it = iter(out)
doc = re.sub(PARA_RE, lambda m: next(_it), doc, flags=re.S)

# 7b) number every display equation in document order, IEEE style: the math is
# centered on a centre tab and the number "(n)" is right-aligned on a right tab,
# both at the two-column geometry. pandoc emits display math as a centred
# <m:oMathPara>; we unwrap it to a bare inline <m:oMath> so tab stops can place it.
# Column inner width = (11520 - 740 - 740 - 400) / 2 = 4820 twips.
EQ_PPR = ('<w:pPr><w:pStyle w:val="PARA"/>'
          '<w:tabs><w:tab w:val="center" w:pos="2410"/>'
          '<w:tab w:val="right" w:pos="4760"/></w:tabs>'
          '<w:spacing w:before="140" w:after="140" w:line="240" w:lineRule="auto"/></w:pPr>')
_eq = [0]
def number_eq(m):
    par = m.group(0)
    if "<m:oMathPara>" not in par:
        return par
    inner = re.search(r"<m:oMath>.*?</m:oMath>", par, re.S)
    if not inner:
        return par
    _eq[0] += 1
    return ("<w:p>" + EQ_PPR + "<w:r><w:tab/></w:r>" + inner.group(0)
            + '<w:r><w:tab/></w:r><w:r><w:t xml:space="preserve">(' + str(_eq[0])
            + ")</w:t></w:r></w:p>")
doc = re.sub(PARA_RE, number_eq, doc, flags=re.S)
print("numbered", _eq[0], "display equations")

# 8) final section properties -> two columns
doc = re.sub(r"<w:sectPr(?![^>]*\bw:rsidSect=\"__)[^>]*>(?:(?!</w:sectPr>).)*</w:sectPr>\s*</w:body>",
             SECT2 + "</w:body>", doc, count=1, flags=re.S)

# 9) scale every image to one column width, preserving aspect ratio
def scale(m):
    cx, cy = int(m.group(1)), int(m.group(2))
    if cx <= COL_EMU_W:
        return m.group(0)
    ncy = int(cy * COL_EMU_W / cx)
    return m.group(0).replace(f'cx="{cx}"', f'cx="{COL_EMU_W}"').replace(f'cy="{cy}"', f'cy="{ncy}"')

doc = re.sub(r'<wp:extent cx="(\d+)" cy="(\d+)"\s*/>', scale, doc)
doc = re.sub(r'<a:ext cx="(\d+)" cy="(\d+)"\s*/>', scale, doc)

open(p, "w", encoding="utf-8").write(doc)

# rezip
if os.path.exists(DST):
    os.remove(DST)
zf = zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED)
for root, _, files in os.walk(WORK):
    for f in files:
        full = os.path.join(root, f)
        zf.write(full, os.path.relpath(full, WORK))
zf.close()
print("wrote", DST)
