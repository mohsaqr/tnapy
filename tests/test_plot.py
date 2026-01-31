"""Tests for TNA plotting functionality.

These tests verify that the plotting functions work correctly
and produce valid matplotlib figures.
"""

import numpy as np
import pandas as pd
import pytest

import tna

# Check if matplotlib is available for tests
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for testing
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

pytestmark = pytest.mark.skipif(
    not HAS_MATPLOTLIB,
    reason="matplotlib not installed"
)


class TestColorPalette:
    """Tests for color palette functions."""

    def test_color_palette_small(self):
        """Test color palette for small number of states."""
        colors = tna.color_palette(5)
        assert len(colors) == 5
        # All should be valid hex colors
        for c in colors:
            assert c.startswith('#')
            assert len(c) == 7

    def test_color_palette_medium(self):
        """Test color palette for medium number of states."""
        colors = tna.color_palette(10)
        assert len(colors) == 10
        # Should use Set3 palette
        for c in colors:
            assert c.startswith('#')

    def test_color_palette_large(self):
        """Test color palette for large number of states."""
        colors = tna.color_palette(20)
        assert len(colors) == 20
        # Should use HCL generation
        for c in colors:
            assert c.startswith('#')

    def test_color_palette_specific(self):
        """Test requesting specific palette."""
        colors_accent = tna.color_palette(5, palette='accent')
        colors_set3 = tna.color_palette(5, palette='set3')
        colors_hcl = tna.color_palette(5, palette='hcl')

        # All should have 5 colors
        assert len(colors_accent) == 5
        assert len(colors_set3) == 5
        assert len(colors_hcl) == 5

        # They should be different
        assert colors_accent != colors_set3
        assert colors_accent != colors_hcl

    def test_color_palette_default(self):
        """Test default palette."""
        colors = tna.color_palette(5, palette='default')
        assert len(colors) == 5
        # Should match DEFAULT_COLORS
        assert colors == tna.DEFAULT_COLORS[:5]

    def test_color_palette_empty(self):
        """Test empty palette."""
        colors = tna.color_palette(0)
        assert colors == []

    def test_color_palette_invalid(self):
        """Test invalid palette name."""
        with pytest.raises(ValueError, match="Unknown palette"):
            tna.color_palette(5, palette='invalid')

    def test_create_color_map(self):
        """Test create_color_map function."""
        labels = ['A', 'B', 'C']
        color_map = tna.create_color_map(labels)

        assert isinstance(color_map, dict)
        assert set(color_map.keys()) == set(labels)
        for color in color_map.values():
            assert color.startswith('#')

    def test_create_color_map_custom(self):
        """Test create_color_map with custom colors."""
        labels = ['A', 'B', 'C']
        custom_colors = ['#ff0000', '#00ff00', '#0000ff']
        color_map = tna.create_color_map(labels, colors=custom_colors)

        assert color_map['A'] == '#ff0000'
        assert color_map['B'] == '#00ff00'
        assert color_map['C'] == '#0000ff'


class TestPlotNetwork:
    """Tests for network plotting."""

    @pytest.fixture
    def simple_model(self):
        """Simple TNA model for testing."""
        df = pd.DataFrame({
            'step1': ['A', 'B', 'A', 'C', 'B'],
            'step2': ['B', 'C', 'B', 'A', 'A'],
            'step3': ['C', 'A', 'C', 'B', 'C'],
        })
        return tna.tna(df)

    def test_plot_network_basic(self, simple_model):
        """Test basic network plot."""
        ax = tna.plot_network(simple_model)

        assert ax is not None
        # Check that we have nodes
        plt.close('all')

    def test_plot_network_layouts(self, simple_model):
        """Test different layout algorithms."""
        layouts = ['circular', 'spring', 'kamada_kawai', 'shell', 'spectral', 'random']

        for layout in layouts:
            ax = tna.plot_network(simple_model, layout=layout)
            assert ax is not None
            plt.close('all')

    def test_plot_network_invalid_layout(self, simple_model):
        """Test invalid layout raises error."""
        with pytest.raises(ValueError, match="Unknown layout"):
            tna.plot_network(simple_model, layout='invalid_layout')

    def test_plot_network_node_size_centrality(self, simple_model):
        """Test node sizing by centrality."""
        ax = tna.plot_network(simple_model, node_size='OutStrength')
        assert ax is not None
        plt.close('all')

    def test_plot_network_node_size_fixed(self, simple_model):
        """Test fixed node size."""
        ax = tna.plot_network(simple_model, node_size=2000)
        assert ax is not None
        plt.close('all')

    def test_plot_network_edge_threshold(self, simple_model):
        """Test edge threshold filtering."""
        ax = tna.plot_network(simple_model, edge_threshold=0.3)
        assert ax is not None
        plt.close('all')

    def test_plot_network_no_self_loops(self, simple_model):
        """Test hiding self-loops."""
        ax = tna.plot_network(simple_model, show_self_loops=False)
        assert ax is not None
        plt.close('all')

    def test_plot_network_no_edge_labels(self, simple_model):
        """Test without edge labels."""
        ax = tna.plot_network(simple_model, edge_labels=False)
        assert ax is not None
        plt.close('all')

    def test_plot_network_custom_colors(self, simple_model):
        """Test custom node colors."""
        colors = ['#ff0000', '#00ff00', '#0000ff']
        ax = tna.plot_network(simple_model, colors=colors)
        assert ax is not None
        plt.close('all')

    def test_plot_network_custom_labels(self, simple_model):
        """Test custom node labels."""
        labels = ['State 1', 'State 2', 'State 3']
        ax = tna.plot_network(simple_model, labels=labels)
        assert ax is not None
        plt.close('all')

    def test_plot_network_with_title(self, simple_model):
        """Test with title."""
        ax = tna.plot_network(simple_model, title='Test Network')
        assert ax is not None
        assert ax.get_title() == 'Test Network'
        plt.close('all')

    def test_plot_network_existing_axes(self, simple_model):
        """Test plotting on existing axes."""
        fig, ax = plt.subplots()
        result_ax = tna.plot_network(simple_model, ax=ax)
        assert result_ax is ax
        plt.close('all')


class TestPlotCentralities:
    """Tests for centrality plotting."""

    @pytest.fixture
    def centralities(self):
        """Centralities DataFrame for testing."""
        df = pd.DataFrame({
            'step1': ['A', 'B', 'A', 'C', 'B'],
            'step2': ['B', 'C', 'B', 'A', 'A'],
            'step3': ['C', 'A', 'C', 'B', 'C'],
        })
        model = tna.tna(df)
        return tna.centralities(model)

    def test_plot_centralities_basic(self, centralities):
        """Test basic centrality plot."""
        fig = tna.plot_centralities(centralities)
        assert fig is not None
        plt.close('all')

    def test_plot_centralities_specific_measures(self, centralities):
        """Test plotting specific measures."""
        fig = tna.plot_centralities(centralities, measures=['InStrength', 'OutStrength'])
        assert fig is not None
        plt.close('all')

    def test_plot_centralities_invalid_measure(self, centralities):
        """Test error on invalid measure."""
        with pytest.raises(ValueError, match="Unknown measures"):
            tna.plot_centralities(centralities, measures=['InvalidMeasure'])

    def test_plot_centralities_ncol(self, centralities):
        """Test different number of columns."""
        fig = tna.plot_centralities(centralities, ncol=2)
        assert fig is not None
        plt.close('all')

    def test_plot_centralities_normalized(self, centralities):
        """Test normalized centralities."""
        fig = tna.plot_centralities(centralities, normalize=True)
        assert fig is not None
        plt.close('all')

    def test_plot_centralities_unsorted(self, centralities):
        """Test unsorted bars."""
        fig = tna.plot_centralities(centralities, sort_values=False)
        assert fig is not None
        plt.close('all')

    def test_plot_centralities_custom_colors(self, centralities):
        """Test custom colors."""
        colors = ['#ff0000', '#00ff00', '#0000ff']
        fig = tna.plot_centralities(centralities, colors=colors)
        assert fig is not None
        plt.close('all')

    def test_plot_centralities_with_title(self, centralities):
        """Test with title."""
        fig = tna.plot_centralities(centralities, title='Test Centralities')
        assert fig is not None
        plt.close('all')


class TestPlotHeatmap:
    """Tests for heatmap plotting."""

    @pytest.fixture
    def simple_model(self):
        """Simple TNA model for testing."""
        df = pd.DataFrame({
            'step1': ['A', 'B', 'A', 'C', 'B'],
            'step2': ['B', 'C', 'B', 'A', 'A'],
            'step3': ['C', 'A', 'C', 'B', 'C'],
        })
        return tna.tna(df)

    def test_plot_heatmap_basic(self, simple_model):
        """Test basic heatmap plot."""
        ax = tna.plot_heatmap(simple_model)
        assert ax is not None
        plt.close('all')

    def test_plot_heatmap_no_annot(self, simple_model):
        """Test heatmap without annotations."""
        ax = tna.plot_heatmap(simple_model, annot=False)
        assert ax is not None
        plt.close('all')

    def test_plot_heatmap_custom_cmap(self, simple_model):
        """Test custom colormap."""
        ax = tna.plot_heatmap(simple_model, cmap='viridis')
        assert ax is not None
        plt.close('all')

    def test_plot_heatmap_custom_fmt(self, simple_model):
        """Test custom format string."""
        ax = tna.plot_heatmap(simple_model, fmt=".3f")
        assert ax is not None
        plt.close('all')

    def test_plot_heatmap_with_title(self, simple_model):
        """Test with title."""
        ax = tna.plot_heatmap(simple_model, title='Test Heatmap')
        assert ax is not None
        plt.close('all')

    def test_plot_heatmap_existing_axes(self, simple_model):
        """Test plotting on existing axes."""
        fig, ax = plt.subplots()
        result_ax = tna.plot_heatmap(simple_model, ax=ax)
        assert result_ax is ax
        plt.close('all')


class TestPlotComparison:
    """Tests for comparison plotting."""

    @pytest.fixture
    def models(self):
        """Two TNA models for comparison."""
        df = pd.DataFrame({
            'step1': ['A', 'B', 'A', 'C', 'B'],
            'step2': ['B', 'C', 'B', 'A', 'A'],
            'step3': ['C', 'A', 'C', 'B', 'C'],
        })
        model1 = tna.tna(df)
        model2 = tna.ftna(df)
        return model1, model2

    def test_plot_comparison_heatmap(self, models):
        """Test heatmap comparison."""
        model1, model2 = models
        fig = tna.plot_comparison(model1, model2, plot_type='heatmap')
        assert fig is not None
        plt.close('all')

    def test_plot_comparison_scatter(self, models):
        """Test scatter comparison."""
        model1, model2 = models
        fig = tna.plot_comparison(model1, model2, plot_type='scatter')
        assert fig is not None
        plt.close('all')

    def test_plot_comparison_network(self, models):
        """Test network comparison."""
        model1, model2 = models
        fig = tna.plot_comparison(model1, model2, plot_type='network')
        assert fig is not None
        plt.close('all')

    def test_plot_comparison_custom_labels(self, models):
        """Test custom labels."""
        model1, model2 = models
        fig = tna.plot_comparison(model1, model2, labels=('TNA', 'FTNA'))
        assert fig is not None
        plt.close('all')

    def test_plot_comparison_invalid_type(self, models):
        """Test invalid plot type."""
        model1, model2 = models
        with pytest.raises(ValueError, match="Unknown plot_type"):
            tna.plot_comparison(model1, model2, plot_type='invalid')

    def test_plot_comparison_different_labels(self):
        """Test error when models have different labels."""
        df1 = pd.DataFrame({
            'step1': ['A', 'B'],
            'step2': ['B', 'A'],
        })
        df2 = pd.DataFrame({
            'step1': ['X', 'Y'],
            'step2': ['Y', 'X'],
        })
        model1 = tna.tna(df1)
        model2 = tna.tna(df2)

        with pytest.raises(ValueError, match="same state labels"):
            tna.plot_comparison(model1, model2)


class TestPlotSequences:
    """Tests for sequence plotting."""

    @pytest.fixture
    def sequence_data(self):
        """Sequence data for testing."""
        return pd.DataFrame({
            'step1': ['A', 'B', 'A', 'C', 'B'],
            'step2': ['B', 'C', 'B', 'A', 'A'],
            'step3': ['C', 'A', 'C', 'B', 'C'],
        })

    def test_plot_sequences_index(self, sequence_data):
        """Test index plot."""
        fig = tna.plot_sequences(sequence_data, plot_type='index')
        assert fig is not None
        plt.close('all')

    def test_plot_sequences_distribution(self, sequence_data):
        """Test distribution plot."""
        fig = tna.plot_sequences(sequence_data, plot_type='distribution')
        assert fig is not None
        plt.close('all')

    def test_plot_sequences_invalid_type(self, sequence_data):
        """Test invalid plot type."""
        with pytest.raises(ValueError, match="Unknown plot_type"):
            tna.plot_sequences(sequence_data, plot_type='invalid')

    def test_plot_sequences_max_sequences(self, sequence_data):
        """Test max_sequences parameter."""
        fig = tna.plot_sequences(sequence_data, plot_type='index', max_sequences=3)
        assert fig is not None
        plt.close('all')

    def test_plot_sequences_custom_colors(self, sequence_data):
        """Test custom colors."""
        colors = ['#ff0000', '#00ff00', '#0000ff']
        fig = tna.plot_sequences(sequence_data, colors=colors)
        assert fig is not None
        plt.close('all')

    def test_plot_sequences_with_title(self, sequence_data):
        """Test with title."""
        fig = tna.plot_sequences(sequence_data, title='Test Sequences')
        assert fig is not None
        plt.close('all')

    def test_plot_sequences_from_tnadata(self):
        """Test plotting from TNAData object."""
        # Create TNAData
        long_data = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u1', 'u2', 'u2', 'u2'],
            'action': ['A', 'B', 'C', 'B', 'A', 'C'],
        })
        prepared = tna.prepare_data(long_data, actor='user_id', action='action')

        fig = tna.plot_sequences(prepared, plot_type='index')
        assert fig is not None
        plt.close('all')


class TestPlotFrequencies:
    """Tests for frequency plotting."""

    @pytest.fixture
    def simple_model(self):
        """Simple TNA model for testing."""
        df = pd.DataFrame({
            'step1': ['A', 'B', 'A', 'C', 'B'],
            'step2': ['B', 'C', 'B', 'A', 'A'],
            'step3': ['C', 'A', 'C', 'B', 'C'],
        })
        return tna.tna(df)

    def test_plot_frequencies_basic(self, simple_model):
        """Test basic frequency plot."""
        ax = tna.plot_frequencies(simple_model)
        assert ax is not None
        plt.close('all')

    def test_plot_frequencies_vertical(self, simple_model):
        """Test vertical bars."""
        ax = tna.plot_frequencies(simple_model, horizontal=False)
        assert ax is not None
        plt.close('all')

    def test_plot_frequencies_unsorted(self, simple_model):
        """Test unsorted bars."""
        ax = tna.plot_frequencies(simple_model, sort_values=False)
        assert ax is not None
        plt.close('all')

    def test_plot_frequencies_custom_colors(self, simple_model):
        """Test custom colors."""
        colors = ['#ff0000', '#00ff00', '#0000ff']
        ax = tna.plot_frequencies(simple_model, colors=colors)
        assert ax is not None
        plt.close('all')

    def test_plot_frequencies_with_title(self, simple_model):
        """Test with title."""
        ax = tna.plot_frequencies(simple_model, title='Test Frequencies')
        assert ax is not None
        plt.close('all')

    def test_plot_frequencies_existing_axes(self, simple_model):
        """Test plotting on existing axes."""
        fig, ax = plt.subplots()
        result_ax = tna.plot_frequencies(simple_model, ax=ax)
        assert result_ax is ax
        plt.close('all')


class TestPlotHistogram:
    """Tests for histogram plotting."""

    @pytest.fixture
    def simple_model(self):
        """Simple TNA model for testing."""
        df = pd.DataFrame({
            'step1': ['A', 'B', 'A', 'C', 'B'],
            'step2': ['B', 'C', 'B', 'A', 'A'],
            'step3': ['C', 'A', 'C', 'B', 'C'],
        })
        return tna.tna(df)

    def test_plot_histogram_basic(self, simple_model):
        """Test basic histogram plot."""
        ax = tna.plot_histogram(simple_model)
        assert ax is not None
        plt.close('all')

    def test_plot_histogram_custom_bins(self, simple_model):
        """Test custom number of bins."""
        ax = tna.plot_histogram(simple_model, bins=10)
        assert ax is not None
        plt.close('all')

    def test_plot_histogram_include_zeros(self, simple_model):
        """Test including zeros."""
        ax = tna.plot_histogram(simple_model, include_zeros=True)
        assert ax is not None
        plt.close('all')

    def test_plot_histogram_custom_color(self, simple_model):
        """Test custom color."""
        ax = tna.plot_histogram(simple_model, color='red')
        assert ax is not None
        plt.close('all')

    def test_plot_histogram_with_title(self, simple_model):
        """Test with title."""
        ax = tna.plot_histogram(simple_model, title='Test Histogram')
        assert ax is not None
        plt.close('all')

    def test_plot_histogram_existing_axes(self, simple_model):
        """Test plotting on existing axes."""
        fig, ax = plt.subplots()
        result_ax = tna.plot_histogram(simple_model, ax=ax)
        assert result_ax is ax
        plt.close('all')


class TestGroupRegulationPlots:
    """Tests with the group regulation dataset."""

    @pytest.fixture
    def group_regulation_model(self):
        """TNA model from group regulation data."""
        df = tna.load_group_regulation()
        return tna.tna(df)

    def test_plot_network_group_regulation(self, group_regulation_model):
        """Test network plot with real data."""
        ax = tna.plot_network(group_regulation_model)
        assert ax is not None
        plt.close('all')

    def test_plot_network_centrality_sizing(self, group_regulation_model):
        """Test network with centrality-based sizing."""
        ax = tna.plot_network(
            group_regulation_model,
            node_size='Betweenness',
            layout='spring'
        )
        assert ax is not None
        plt.close('all')

    def test_plot_centralities_group_regulation(self, group_regulation_model):
        """Test centrality plot with real data."""
        cent = tna.centralities(group_regulation_model)
        fig = tna.plot_centralities(cent)
        assert fig is not None
        plt.close('all')

    def test_plot_heatmap_group_regulation(self, group_regulation_model):
        """Test heatmap with real data."""
        ax = tna.plot_heatmap(group_regulation_model)
        assert ax is not None
        plt.close('all')

    def test_plot_sequences_group_regulation(self):
        """Test sequence plot with real data."""
        df = tna.load_group_regulation()
        fig = tna.plot_sequences(df, max_sequences=50)
        assert fig is not None
        plt.close('all')

    def test_plot_frequencies_group_regulation(self, group_regulation_model):
        """Test frequency plot with real data."""
        ax = tna.plot_frequencies(group_regulation_model)
        assert ax is not None
        plt.close('all')

    def test_full_visualization_workflow(self, group_regulation_model):
        """Test complete visualization workflow."""
        # Network plot
        ax1 = tna.plot_network(group_regulation_model, layout='circular')
        assert ax1 is not None
        plt.close('all')

        # Centrality plot
        cent = tna.centralities(group_regulation_model)
        fig2 = tna.plot_centralities(cent, measures=['OutStrength', 'InStrength', 'Betweenness'])
        assert fig2 is not None
        plt.close('all')

        # Heatmap
        ax3 = tna.plot_heatmap(group_regulation_model)
        assert ax3 is not None
        plt.close('all')

        # Comparison
        model2 = tna.ftna(tna.load_group_regulation())
        fig4 = tna.plot_comparison(group_regulation_model, model2)
        assert fig4 is not None
        plt.close('all')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
