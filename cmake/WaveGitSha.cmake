# Resolves the commit at build time and writes it into a generated header.
#
# Capturing this with a plain execute_process() at configure time is a trap: the
# value then goes stale the moment you commit and rebuild without reconfiguring,
# and the benchmark happily stamps published CSV rows with a commit that does not
# contain the code that produced them. Running here, from a custom target that
# fires every build, keeps the stamp honest.
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY "${WAVE_SOURCE_DIR}"
    OUTPUT_VARIABLE WAVE_GIT_SHA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT WAVE_GIT_SHA)
    set(WAVE_GIT_SHA "unknown")
endif()

# Tracked-file modifications mean the binary does not correspond to the commit,
# which matters when the number ends up in a published measurement.
execute_process(
    COMMAND git status --porcelain --untracked-files=no
    WORKING_DIRECTORY "${WAVE_SOURCE_DIR}"
    OUTPUT_VARIABLE WAVE_GIT_DIRTY
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT WAVE_GIT_DIRTY STREQUAL "")
    set(WAVE_GIT_SHA "${WAVE_GIT_SHA}-dirty")
endif()

configure_file("${WAVE_INPUT}" "${WAVE_OUTPUT}" @ONLY)
