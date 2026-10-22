# Per-repository sandbox image. Network is used only while building this image;
# task containers run with --network none.
FROM python:3.12-slim

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends git patch procps \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -q pytest pytest-timeout==2.1.0 setuptools wheel \
       hatchling hatch-fancy-pypi-readme hatch-vcs poetry-core flit-core pdm-backend editables

RUN git config --global user.email "agent@sandbox" \
    && git config --global user.name "Agent" \
    && git config --global init.defaultBranch main \
    && git config --global --add safe.directory '*'

# The seed checkout pulls in the repo's runtime and test dependencies, then is removed.
COPY seed /opt/seed
COPY pins.txt /opt/pins.txt
RUN cd /opt/seed \
    && (pip install --no-cache-dir -q -e ".[all]" || pip install --no-cache-dir -q -e .) \
    && for f in $(ls requirements*.txt requirements/*.txt 2>/dev/null | grep -viE 'doc|lint|release'); do \
         grep -vE '^\s*-e\s|^\s*-r\s' "$f" > /tmp/req.txt; pip install --no-cache-dir -q -r /tmp/req.txt || true; \
       done \
    && pip uninstall -y -q "$(python -c "import tomllib;print(tomllib.load(open('pyproject.toml','rb')).get('project',{}).get('name',''))" 2>/dev/null || echo none)" || true \
    && pip install --no-cache-dir -q pytest-timeout==2.1.0 -r /opt/pins.txt \
    && rm -rf /opt/seed

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/src:/workspace \
    TEST_TMPDIR=/tmp

WORKDIR /workspace
