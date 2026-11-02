# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check reported addresses, buffer copies and segment-boundary clipping."""

from copy import deepcopy
import importlib
import re
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest

from tensilelite.Common.GlobalParameters import globalParameters, _assertGlobalParametersAreValid
from tensilelite.Common.TypeValidationErrors import ConfigTypeError
from tensilelite.SolutionStructs.lds_layout import format_lds_layout, lds_layout_model, _rotation_model
from tensilelite.SolutionStructs.lds_layout_svg import build_svg, svg_file_name, write_layout_svg
from tensilelite.SolutionStructs.segment_interleave import evaluate, paired_buffer_offsets
from tensilelite.Tests.unit.test_segment_interleave import _vw8_state, _brot41_state, _brot14ab_state

pytestmark = pytest.mark.unit


def _ordinary(**overrides):
    state = dict(ISA=(12, 5, 0), MIWaveGroup=[2, 2], PrefetchGlobalRead=1,
                 NumLdsBlk=2, LdsOffsetA=0, LdsOffsetB=1024, LdsOffsetA_Blk=65536,
                 ldsNumBytesA=1024, ldsNumBytesB=2048, LdsNumBytes=68608)
    state.update(overrides)
    return state


def test_ordinary_addresses_and_segment_order():
    text = format_lds_layout(_ordinary(), 'example')
    assert text.startswith('[LDS layout] example\n')
    assert '[     0,   1024) A ping' in text
    assert '[ 65536,  66560) A pong' in text
    assert 'S0 [0, 65536): [A ping][B ping][gap 62464 B]' in text
    assert 'S1 [65536, 68608): [A pong][B pong]' in text
    segments = [line for line in text.splitlines() if line.startswith('    S')]
    assert len(segments) == 2
    assert segments[0].startswith('    S0 ') and segments[1].startswith('    S1 ')


@pytest.mark.parametrize('overrides', [{'1LDSBuffer': 1}, {'PrefetchGlobalRead': 0}])
def test_single_buffer_does_not_report_phantom_pong(overrides):
    text = format_lds_layout(_ordinary(**overrides), 'single')
    assert 'A buffer0' in text
    assert 'pong' not in text
    assert 'Buffer strides (bytes): A=0, B=0' in text


def test_regions_crossing_segments_appear_in_both():
    text = format_lds_layout(_ordinary(**{'1LDSBuffer': 1, 'ldsNumBytesA': 66048,
                                       'ldsNumBytesB': 0}), 'crossing')
    assert 'S0 [0, 65536): [A buffer0]' in text
    assert 'S1 [65536, 68608): [A buffer0][gap 2560 B]' in text


def test_three_buffers_and_non_gfx1250_have_no_invented_segment_geometry():
    text = format_lds_layout(_ordinary(ISA=(9, 4, 2), NumLdsBlk=3,
                                      LdsNumBytes=134144), 'triple')
    assert '[131072, 132096) A buffer2' in text
    assert '64 KiB segments' not in text


def test_decoupled_groups_use_their_own_buffer_counts_and_strides():
    text = format_lds_layout(_ordinary(PrefetchGlobalReadA=1, PrefetchGlobalReadB=2,
                                      LdsOffsetB=1024, LdsOffsetBlkA=1024,
                                      LdsOffsetBlkB=2048, LdsNumBytes=5120), 'decoupled')
    assert 'A buffer0' in text and 'A pong' not in text
    assert '[  3072,   5120) B pong' in text
    assert 'A=0, B=2048' in text


def _interleaved(**overrides):
    state = _vw8_state(PrefetchGlobalRead=2, LDSSegmentInterleave=1,
                       NumLdsBlk=2, MaxLDS=327680, LdsOffsetA_Blk=131072,
                       VectorWidthB=4, **overrides)
    state['LDSSegInterleaveOffsets'] = evaluate(state)['offsets']
    return state


def test_paired_physical_order_scales_and_no_state_mutation():
    state = _interleaved(ProblemType=dict(MXBlockA=32, MXBlockB=32),
                         LdsNumElementsAlignedMXSA=1024, LdsNumElementsAlignedMXSB=2048)
    offsets = paired_buffer_offsets(state)
    state.update(LDSSegInterleaveOffsets=offsets, LdsNumBytes=offsets['pairedUsedBytes'])
    before = deepcopy(state)
    text = format_lds_layout(state, 'paired')
    assert 'layout=paired' in text
    names = [line.split(') ')[1].split(' (')[0] for line in text.splitlines()
             if line.startswith('    [')]
    assert names == ['A0 ping', 'A0 pong', 'B0 ping', 'B0 pong',
                     'A1 ping', 'A1 pong', 'B1 ping', 'B1 pong',
                     'MXSA ping', 'MXSA pong', 'MXSB ping', 'MXSB pong']
    assert f"[{offsets['ldsBaseMXSA']:6d}, {offsets['ldsBaseMXSA'] + 1024:6d}) MXSA ping" in text
    assert text == format_lds_layout(state, 'paired')
    # Ignore identity-only fake data types; all scalar/layout fields stay unchanged.
    assert {k: v for k, v in state.items() if k != 'ProblemType'} == {
        k: v for k, v in before.items() if k != 'ProblemType'}


def test_aligned_and_tdmsplit_keep_whole_buffer_stride():
    state = _interleaved(DepthU=64, TDMSplit=1)
    state['LdsNumBytes'] = 229632
    text = format_lds_layout(state, 'tdmsplit')
    assert 'layout=interleaved' in text
    assert '[     0,  16512) A group0 ping' in text
    assert '[ 65536,  82048) A group1 ping' in text
    assert '[131072, 147584) A group0 pong' in text
    assert 'group0/1 name physical read groups' in text


def test_whole_operand_and_direct_scales_are_not_split_or_allocated():
    state = _interleaved(MIWaveTile=[8, 7], VectorWidthA=8,
                         DirectToVgprMXSA=True, LdsNumElementsAlignedMXSA=2048)
    # Force B's whole-operand fallback (the helper otherwise uses VWB=4).
    state['VectorWidthB'] = 1
    state['LDSSegInterleaveOffsets'] = evaluate(state)['offsets']
    offsets = paired_buffer_offsets(state)
    state.update(LDSSegInterleaveOffsets=offsets, LdsNumBytes=offsets['pairedUsedBytes'])
    text = format_lds_layout(state, 'whole')
    assert 'B ping' in text and 'B0 ping' not in text
    assert 'MXSA ping' not in text


def test_global_flag_is_boolean_and_off_by_default():
    assert globalParameters['PrintLDSLayout'] is False
    _assertGlobalParametersAreValid({'PrintLDSLayout': True}, [])
    with pytest.raises(ConfigTypeError):
        _assertGlobalParametersAreValid({'PrintLDSLayout': 1}, [])


@pytest.mark.parametrize('enabled,valid,expected', [(False, True, False),
                                                   (True, False, False),
                                                   (True, True, True)])
def test_solution_logs_only_valid_final_layout_when_enabled(monkeypatch, capsys,
                                                          enabled, valid, expected):
    module = importlib.import_module('tensilelite.SolutionStructs.Solution')
    final = _ordinary(Valid=valid)
    # Bypass hardware/toolchain derivation; exercise the real constructor and
    # log with a fully specified allocation, independent of local GPU support.
    monkeypatch.setattr(module.Solution, 'assignDerivedParameters',
                        staticmethod(lambda state, *args: state.update(final)))
    monkeypatch.setattr(module, 'getSolutionNameFull', lambda *args: 'test_kernel')
    monkeypatch.setitem(globalParameters, 'PrintLDSLayout', enabled)
    isa = (12, 5, 0)
    module.Solution({'ISA': isa, 'ProblemType': {'OperationType': 'GEMM', 'DataType': 'H'}},
                    False, False, False,
                    SimpleNamespace(code_object_version='default', rocm_version=None),
                    {isa: SimpleNamespace(asmCaps={'SupportedISA': True})})
    text = capsys.readouterr().out
    assert ('[LDS layout] test_kernel' in text) == expected
    if expected:
        assert '[ 65536,  66560) A pong' in text


def _asymmetric_layout(wg, variant):
    active = 'A' if wg == [4, 1] else 'B'
    tile = [4, 8] if active == 'A' else [8, 4]
    widths = [4, 8] if active == 'A' else [8, 4]
    split = variant in ('tdmsplit_full', 'tdmsplit_half')
    if variant == 'tdmsplit_half':
        widths[0 if active == 'A' else 1] //= 2
    state = _vw8_state(MIWaveGroup=wg, MIWaveTile=tile,
                       VectorWidthA=widths[0], VectorWidthB=widths[1],
                       DepthU=64 if variant == 'aligned' else 128,
                       TDMSplit=split, PrefetchGlobalRead=2,
                       LDSSegmentInterleave=1, NumLdsBlk=2,
                       LdsOffsetA_Blk=131072, MaxLDS=327680)
    result = evaluate(state)
    assert result['applicable'], result['reason']
    state['LDSSegInterleaveOffsets'] = result['offsets']
    if variant == 'paired':
        offsets = paired_buffer_offsets(state)
        assert offsets is not None
        state.update(LDSSegInterleaveOffsets=offsets, LdsNumBytes=offsets['pairedUsedBytes'])
    else:
        # The complete ping block fits a 128 KiB stride. Its last active group
        # ends at 82048 bytes (aligned) or 99072 bytes (TDMSplit).
        state['LdsNumBytes'] = 131072 + (82048 if variant == 'aligned' else 99072)
    return state


@pytest.mark.parametrize('wg,active,whole', [([4, 1], 'A', 'B'), ([1, 4], 'B', 'A')])
@pytest.mark.parametrize('variant', ['paired', 'aligned', 'tdmsplit_full', 'tdmsplit_half'])
def test_asymmetric_patterns_report_active_groups_and_whole_operand(wg, active, whole, variant):
    state = _asymmetric_layout(wg, variant)
    text = format_lds_layout(state, f'{wg}_{variant}')
    regions = [line for line in text.splitlines() if line.startswith('    [')]
    assert len(regions) == 6
    assert 'MIWaveGroup=' + str(wg) in text
    assert f'{whole} ping' in text and f'{whole} pong' in text
    assert f'{whole}0 ping' not in text and f'{whole} group0 ping' not in text
    if variant == 'paired':
        assert 'layout=paired' in text
        names = [line.split(') ')[1].split(' (')[0] for line in regions]
        assert names == [f'{active}0 ping', f'{active}0 pong', f'{whole} ping',
                         f'{whole} pong', f'{active}1 ping', f'{active}1 pong']
        assert f'[132096, 165120) {active}1 ping' in text
        assert f'[ 66048,  99072) {whole} ping' in text
    elif variant == 'aligned':
        assert f'[ 65536,  82048) {active}1 ping' in text
        assert f'[196608, 213120) {active}1 pong' in text
        assert f'[ 16512,  33024) {whole} ping' in text
    else:
        assert f'[ 66048,  99072) {active} group1 ping' in text
        assert f'[ 33024,  66048) {whole} ping' in text
        offsets = state['LDSSegInterleaveOffsets']
        key = 'componentSplit' if variant == 'tdmsplit_half' else 'portSplit' + active
        assert offsets[key]


# ---- rotation read order, bars and SVG ----

def _rotating_41_tight():
    state = _brot41_state(tight=True, MaxLDS=5 * 65536)
    offsets = paired_buffer_offsets(state)
    state.update(LDSSegInterleaveOffsets=offsets, LdsNumBytes=offsets['pairedUsedBytes'])
    return state


def _rotating_14_ab():
    state = _brot14ab_state()
    state['LdsNumBytes'] = 131072 + evaluate(state)['blockSpan']
    return state


def _segs(region):
    return set(range(region[0] // 65536, (region[1] - 1) // 65536 + 1))


@pytest.mark.parametrize('make_state,tensors', [(_rotating_41_tight, ['B']), (_rotating_14_ab, ['A', 'B'])])
def test_rotation_model_follows_the_s_xor_p_rule_and_is_conflict_free(make_state, tensors):
    model = lds_layout_model(make_state(), 'k')
    assert sorted({e['tensor'] for e in model['rotations']}) == tensors
    assert len(model['rotations']) == 2 * len(tensors)        # ping and pong per rotated tensor
    for entry in model['rotations']:
        by_comp = {r[4]: r for r in model['regions'] if r[3] == entry['tensor'] and r[5] == entry['phase']}
        for st in entry['steps']:
            assert st['pair0'][0] == st['step'] and st['pair1'][0] == st['step'] ^ 1
            assert st['pair0'][1] == tuple(sorted(_segs(by_comp[st['pair0'][0]])))
            assert st['ok']


def test_rotation_model_reports_a_conflict_when_components_share_a_segment():
    model = dict(gfx1250=True, rotate=(False, True),
                 regions=[(0, 1000, 'B0 ping', 'B', 0, 'ping'), (2000, 3000, 'B1 ping', 'B', 1, 'ping')])
    steps = _rotation_model(model)[0]['steps']
    assert [st['ok'] for st in steps] == [False, False]
    assert 'CONFLICT' in build_svg(dict(name='x', macro_tile=(None, None), wave_group=[1, 4], layout='paired',
                                        rotate=(False, True), reserved=65536, has_offsets=True,
                                        regions=model['regions'], rotations=_rotation_model(model)))


def test_rotation_model_is_absent_without_rotation_and_off_gfx1250():
    assert lds_layout_model(_ordinary(), 'k')['rotations'] == []
    state = _rotating_41_tight()
    state['ISA'] = (9, 4, 2)
    assert lds_layout_model(state, 'k')['rotations'] == []


def test_text_has_rotation_table_and_bars_without_ansi_by_default():
    text = format_lds_layout(_rotating_41_tight(), 'k')
    assert 'Rotation B ping' in text and 'Rotation B pong' in text
    assert 'pair0 B0' in text and 'pair1 B1' in text and 'CONFLICT' not in text
    assert '\033[' not in text
    bars = [line for line in text.splitlines() if line.startswith('  seg ')]
    assert len(bars) == 5
    assert all(len(line.split('|')[1]) == 64 for line in bars)
    assert 'A' in bars[0] and 'b' in bars[1] + bars[2] + bars[3]
    # existing line formats used by older assertions stay disjoint from the new sections
    assert not any(line.startswith('    [') or line.startswith('    S') for line in text.splitlines()
                   if line.startswith('  seg ') or line.startswith('    step'))


def test_bars_can_be_colored_explicitly():
    text = format_lds_layout(_rotating_41_tight(), 'k', color=True)
    assert '\033[34mA\033[0m' in text
    plain = re.sub(r'\033\[[0-9;]*m', '', text)
    assert plain == format_lds_layout(_rotating_41_tight(), 'k', color=False)


def test_non_rotating_layout_prints_bars_but_no_rotation_section():
    text = format_lds_layout(_ordinary(), 'k')
    assert '  seg 0 |' in text and 'Rotation' not in text


@pytest.mark.parametrize('make_state', [_rotating_41_tight, _rotating_14_ab])
def test_svg_is_well_formed_and_names_every_region(make_state):
    state = make_state()
    model = lds_layout_model(state, 'my_kernel')
    svg = build_svg(model)
    root = ET.fromstring(svg)
    ns = {'s': 'http://www.w3.org/2000/svg'}
    assert root.find('s:title', ns).text == 'my_kernel'
    texts = [t.text or '' for t in root.iter('{http://www.w3.org/2000/svg}text')]
    for start, end, name, *_ in model['regions']:
        assert f'[{start:7d}, {end:7d})  {name}  ({end - start} B)' in texts
    text = ' '.join(texts)
    assert 'no conflict' in text and 'CONFLICT' not in text
    assert 'Rotation read order' in text


def test_svg_file_name_is_short_stable_and_unique_per_kernel(tmp_path):
    state = _rotating_41_tight()
    long_name = 'Cijk_' + 'x' * 600
    path = write_layout_svg(state, long_name, str(tmp_path / 'sub'))
    assert path == write_layout_svg(state, long_name, str(tmp_path / 'sub'))
    name = path.rsplit('/', 1)[1]
    assert len(name) < 100 and name.startswith('MT') and name.endswith('.svg')
    assert name != svg_file_name(lds_layout_model(state, long_name + 'y'))
    ET.parse(path)


def test_formatting_and_svg_do_not_mutate_state(tmp_path):
    state = _rotating_41_tight()
    before = deepcopy({k: v for k, v in state.items() if k != 'ProblemType'})
    format_lds_layout(state, 'k')
    write_layout_svg(state, 'k', str(tmp_path))
    assert {k: v for k, v in state.items() if k != 'ProblemType'} == before


def test_svg_dir_global_defaults_to_off():
    assert globalParameters['LDSLayoutDir'] is None


@pytest.mark.parametrize('directory,expected', [(None, 0), ('given', 1)])
def test_solution_writes_svg_only_when_a_directory_is_set(monkeypatch, tmp_path, directory, expected):
    module = importlib.import_module('tensilelite.SolutionStructs.Solution')
    final = _rotating_41_tight()
    final.update(Valid=True)
    monkeypatch.setattr(module.Solution, 'assignDerivedParameters',
                        staticmethod(lambda state, *args: state.update(final)))
    monkeypatch.setattr(module, 'getSolutionNameFull', lambda *args: 'test_kernel')
    monkeypatch.setitem(globalParameters, 'PrintLDSLayout', True)
    monkeypatch.setitem(globalParameters, 'LDSLayoutDir', str(tmp_path / 'lds') if directory else None)
    isa = (12, 5, 0)
    module.Solution({'ISA': isa, 'ProblemType': {'OperationType': 'GEMM', 'DataType': 'H'}},
                    False, False, False,
                    SimpleNamespace(code_object_version='default', rocm_version=None),
                    {isa: SimpleNamespace(asmCaps={'SupportedISA': True})})
    files = list((tmp_path / 'lds').glob('*.svg')) if directory else []
    assert len(files) == expected
    assert (tmp_path / 'lds').exists() == bool(directory)
