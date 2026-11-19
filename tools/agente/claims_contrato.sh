#!/usr/bin/env bash
# Antes do push: claim ativo em CLAIMS.yaml x contrato que exige lista vazia (L-013).
# Saidas: 0 ok | 8 PARAR (claim vai derrubar o CI) | 3 entrada invalida.
# PyYAML e o caminho normal. O job Agent guard replay nao instala dependencias,
# a imagem ubuntu-24.04 nao tem python3-yaml e o workflow nao e editado.
# Sem PyYAML entra um subconjunto estrito, conferido contra PyYAML nos
# CLAIMS.yaml historicos desta guarda; sintaxe fora dele sai 3 (fechado).
set -uo pipefail
dir=${1:-.}; PY=${PY:-python3}
[[ -d $dir && -f $dir/docs/agentes/CLAIMS.yaml ]] || { echo "ENTRADA INVALIDA: $dir sem docs/agentes/CLAIMS.yaml" >&2; exit 3; }
"$PY" - "$dir" <<'PYEOF'
import pathlib, re, sys
try:
    import yaml
except ImportError:
    yaml = None

class _YErr(Exception):
    pass

def _strip_comment(s):
    out = []
    q = None
    for c in s:
        if q:
            out.append(c)
            if c == q:
                q = None
            continue
        if c in ('"', "'"):
            q = c
            out.append(c)
            continue
        if c == '#':
            break
        out.append(c)
    return ''.join(out).rstrip()

def _parse_scalar(s):
    s = s.strip()
    if s in ('|', '>', '|-', '>+', '|+', '>-'):
        raise _YErr('block scalar')
    if s == '[]':
        return []
    if s == '{}':
        return {}
    if s in ('null', '~', 'Null', 'NULL'):
        return None
    if s in ('true', 'True', 'TRUE'):
        return True
    if s in ('false', 'False', 'FALSE'):
        return False
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        body = s[1:-1]
        if s[0] == '"' and '\\' in body:
            raise _YErr('escape')
        return body
    if s[:1] in '[{"\'&*!|>' or s.startswith('---'):
        raise _YErr('flow or special')
    if re.search(r':( |$)', s):
        raise _YErr('bare colon')
    if re.fullmatch(r'-?\d+', s):
        return int(s)
    if re.fullmatch(r'-?\d+\.\d+', s):
        return float(s)
    return s

def _tokenize(text):
    lines = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        raw = raw.replace('\r', '')
        if '\t' in raw:
            raise _YErr(f'tab {lineno}')
        line = _strip_comment(raw)
        if line.strip() == '':
            continue
        indent = len(line) - len(line.lstrip(' '))
        content = line.strip()
        if content in ('---', '...'):
            raise _YErr('marker')
        lines.append((indent, content, lineno))
    return lines

def _key_rest(content):
    m = re.match(r'^([^:]+):(.*)$', content)
    if not m:
        raise _YErr('not a key')
    key = m.group(1).strip()
    if not key or key[0] in ('"', "'", '[', '{'):
        raise _YErr('bad key')
    return key, m.group(2).strip()

def _parse_node(lines, i):
    _indent, content, _lineno = lines[i]
    if content.startswith('- '):
        return _parse_list(lines, i, _indent)
    if re.search(r':( |$)', content):
        return _parse_map(lines, i, _indent)
    if i + 1 != len(lines):
        raise _YErr('scalar doc')
    return _parse_scalar(content), i + 1

def _parse_map(lines, i, indent):
    node = {}
    while i < len(lines):
        ind, content, lineno = lines[i]
        if ind < indent:
            break
        if ind > indent:
            raise _YErr(f'nest {lineno}')
        if content.startswith('- '):
            break
        if not re.search(r':( |$)', content):
            raise _YErr(content)
        key, rest = _key_rest(content)
        i += 1
        if rest == '':
            if i < len(lines) and lines[i][0] > indent:
                child, i = _parse_node(lines, i)
                node[key] = child
            else:
                node[key] = None
        else:
            node[key] = _parse_scalar(rest)
    return node, i

def _parse_list(lines, i, indent):
    items = []
    while i < len(lines) and lines[i][0] == indent and lines[i][1].startswith('- '):
        _, content, _lineno = lines[i]
        item = content[2:].strip()
        i += 1
        if item == '':
            raise _YErr('empty item')
        if re.search(r':( |$)', item) and item[0] not in ('"', "'"):
            node = {}

            def eat(text, idx, base, node=node):
                key, rest = _key_rest(text)
                if rest == '':
                    if idx < len(lines) and lines[idx][0] > base:
                        child, idx = _parse_node(lines, idx)
                        node[key] = child
                    else:
                        node[key] = None
                else:
                    node[key] = _parse_scalar(rest)
                return idx

            i = eat(item, i, indent)
            while i < len(lines) and lines[i][0] > indent:
                if lines[i][1].startswith('- '):
                    raise _YErr('orphan dash')
                ind2, content2, _ = lines[i]
                i += 1
                i = eat(content2, i, ind2)
            items.append(node)
        else:
            if i < len(lines) and lines[i][0] > indent:
                raise _YErr('indent under scalar')
            items.append(_parse_scalar(item))
    return items, i

def _subset_yaml(text):
    lines = _tokenize(text)
    if not lines:
        return None
    val, i = _parse_node(lines, 0)
    if i != len(lines):
        raise _YErr('trailing')
    return val

def _load(text):
    if yaml is not None:
        return yaml.safe_load(text)
    return _subset_yaml(text)

root = pathlib.Path(sys.argv[1])
try:
    data = _load((root / "docs/agentes/CLAIMS.yaml").read_text(encoding="utf-8"))
    claims = data["claims"]
    assert claims is None or isinstance(claims, list)
except Exception as e:
    print(f"ENTRADA INVALIDA: CLAIMS.yaml ilegivel ({type(e).__name__})", file=sys.stderr); sys.exit(3)
if not claims:
    print("OK: sem claims ativos"); sys.exit(0)
pat = re.compile(r"""assert\w*\(\s*["']claims: \[\]["']|\[\s*["']claims["']\s*\]\s*,\s*\[\]""")
hits = sorted(str(p.relative_to(root)) for p in root.rglob("test_*.py")
              if ".git" not in p.parts and pat.search(p.read_text(encoding="utf-8", errors="replace")))
if hits:
    print(f"PARAR: {len(claims)} claim(s) ativo(s) e contrato exige lista vazia em: {' '.join(hits)} "
          "(L-013; o CI vai falhar). Peca decisao; nao edite o teste."); sys.exit(8)
print(f"OK: {len(claims)} claim(s) ativo(s), nenhum contrato de lista vazia"); sys.exit(0)
PYEOF
