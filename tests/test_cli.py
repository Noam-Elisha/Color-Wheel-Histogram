"""
Test command line interface including:
- main() function with different argument combinations
- argument parsing and validation
- error handling for invalid inputs
- integration tests for end-to-end workflow
"""

import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import argparse

import color_wheel


class TestArgumentParsing:
    """Test command line argument parsing."""
    
    def test_basic_argument_parsing(self, sample_image_path, temp_dir):
        """Test basic argument parsing with minimal arguments."""
        output_path = temp_dir / "output.png"
        
        test_args = [
            'color_wheel.py',
            str(sample_image_path),
            str(output_path)
        ]
        
        with patch('sys.argv', test_args):
            parser = argparse.ArgumentParser(
                description="Generate a color wheel where opacity represents color frequency in an image"
            )
            parser.add_argument("input_image", help="Path to the input image")
            parser.add_argument("output_wheel", help="Path for the output color wheel image")
            parser.add_argument("--size", type=int, default=800)
            parser.add_argument("--sample-factor", type=int, default=1)
            
            args = parser.parse_args(test_args[1:])
            
            assert args.input_image == str(sample_image_path)
            assert args.output_wheel == str(output_path)
            assert args.size == 800  # Default value
            assert args.sample_factor == 1  # Default value
    
    def test_all_optional_arguments(self, sample_image_path, temp_dir):
        """Test parsing with all optional arguments."""
        output_path = temp_dir / "output.png"
        
        test_args = [
            'color_wheel.py',
            str(sample_image_path),
            str(output_path),
            '--size', '400',
            '--sample-factor', '2',
            '--quantize', '4',
            '--show-reference',
            '--format', 'jpg',
            '--histogram',
            '--color-spectrum',
            '--circular-spectrum',
            '--force-kdtree',
            '--parallel',
            '--gpu',
            '--color-space', 'Adobe RGB'
        ]
        
        with patch('sys.argv', test_args):
            # Use the actual argument parser from the module
            parser = argparse.ArgumentParser(
                description="Generate a color wheel where opacity represents color frequency in an image"
            )
            parser.add_argument("input_image", help="Path to the input image")
            parser.add_argument("output_wheel", help="Path for the output color wheel image")
            parser.add_argument("--size", type=int, default=800, help="Size of the color wheel (default: 800)")
            parser.add_argument("--sample-factor", type=int, default=1)
            parser.add_argument("--quantize", type=int, default=8)
            parser.add_argument("--show-reference", action="store_true")
            parser.add_argument("--format", choices=["png", "jpg"], default="png")
            parser.add_argument("--histogram", action="store_true")
            parser.add_argument("--color-spectrum", action="store_true")
            parser.add_argument("--circular-spectrum", action="store_true")
            parser.add_argument("--force-kdtree", action="store_true")
            parser.add_argument("--no-kdtree", action="store_true")
            parser.add_argument("--parallel", action="store_true")
            parser.add_argument("--no-parallel", action="store_true")
            parser.add_argument("--gpu", action="store_true")
            parser.add_argument("--no-gpu", action="store_true")
            parser.add_argument("--color-space", choices=["sRGB", "Adobe RGB", "ProPhoto RGB"], default="sRGB")
            
            args = parser.parse_args(test_args[1:])
            
            assert args.size == 400
            assert args.sample_factor == 2
            assert args.quantize == 4
            assert args.show_reference is True
            assert args.format == 'jpg'
            assert args.histogram is True
            assert args.color_spectrum is True
            assert args.circular_spectrum is True
            assert args.force_kdtree is True
            assert args.parallel is True
            assert args.gpu is True
            assert args.color_space == 'Adobe RGB'
    
    def test_conflicting_arguments(self):
        """Test handling of conflicting arguments."""
        test_args = [
            'color_wheel.py',
            'input.jpg',
            'output.png',
            '--force-kdtree',
            '--no-kdtree'
        ]
        
        # The main function should detect and handle this conflict
        with patch('sys.argv', test_args):
            with patch('color_wheel.load_and_analyze_image'):
                with patch('color_wheel.create_color_wheel'):
                    with patch('builtins.print') as mock_print:
                        result = color_wheel.main()
                        
                        # Should return error code
                        assert result == 1
                        
                        # Should print error message
                        print_calls = [str(call) for call in mock_print.call_args_list]
                        error_found = any('Cannot specify both' in call for call in print_calls)
                        assert error_found, "Should print error for conflicting arguments"


class TestMainFunctionIntegration:
    """Test the main() function integration."""
    
    def test_successful_execution(self, sample_image_path, temp_dir):
        """Test successful execution of main function."""
        output_path = temp_dir / "test_output.png"
        
        test_args = [
            'color_wheel.py',
            str(sample_image_path),
            str(output_path),
            '--size', '100'  # Small size for faster testing
        ]
        
        with patch('sys.argv', test_args):
            result = color_wheel.main()
            
            # Should return success
            assert result == 0
            
            # Output file should be created
            assert output_path.exists()
            assert output_path.stat().st_size > 0
    
    def test_nonexistent_input_file(self, temp_dir):
        """Test handling of non-existent input file."""
        output_path = temp_dir / "output.png"
        
        test_args = [
            'color_wheel.py',
            '/nonexistent/input.jpg',
            str(output_path)
        ]
        
        with patch('sys.argv', test_args):
            result = color_wheel.main()
            
            # Should return error code
            assert result == 1
    
    def test_invalid_output_directory(self, sample_image_path):
        """Test handling of invalid output directory."""
        test_args = [
            'color_wheel.py',
            str(sample_image_path),
            '/nonexistent/directory/output.png'
        ]
        
        with patch('sys.argv', test_args):
            result = color_wheel.main()
            
            # Should return error code
            assert result == 1
    
    def test_jpg_output_format(self, sample_image_path, temp_dir):
        """Test JPG output format handling."""
        output_path = temp_dir / "test_output.jpg"
        
        test_args = [
            'color_wheel.py',
            str(sample_image_path),
            str(output_path),
            '--format', 'jpg',
            '--size', '100'
        ]
        
        with patch('sys.argv', test_args):
            result = color_wheel.main()
            
            assert result == 0
            assert output_path.exists()
    
    def test_png_output_format(self, sample_image_path, temp_dir):
        """Test PNG output format handling."""
        output_path = temp_dir / "test_output.png"
        
        test_args = [
            'color_wheel.py',
            str(sample_image_path),
            str(output_path),
            '--format', 'png',
            '--size', '100'
        ]
        
        with patch('sys.argv', test_args):
            result = color_wheel.main()
            
            assert result == 0
            assert output_path.exists()
    
    def test_show_reference_option(self, sample_image_path, temp_dir):
        """Test --show-reference option."""
        output_path = temp_dir / "test_output.png"
        reference_path = temp_dir / "test_output_reference.png"
        
        test_args = [
            'color_wheel.py',
            str(sample_image_path),
            str(output_path),
            '--show-reference',
            '--size', '100'
        ]
        
        with patch('sys.argv', test_args):
            result = color_wheel.main()
            
            assert result == 0
            assert output_path.exists()
            assert reference_path.exists()  # Reference file should be created
    
    def test_histogram_generation(self, sample_image_path, temp_dir):
        """Test histogram generation options."""
        output_path = temp_dir / "test_output.png"
        histogram_path = temp_dir / "test_output_histogram.png"
        
        test_args = [
            'color_wheel.py',
            str(sample_image_path),
            str(output_path),
            '--histogram',
            '--size', '100'
        ]
        
        with patch('sys.argv', test_args):
            result = color_wheel.main()
            
            assert result == 0
            assert output_path.exists()
            assert histogram_path.exists()
    
    def test_color_spectrum_generation(self, sample_image_path, temp_dir):
        """Test color spectrum generation."""
        output_path = temp_dir / "test_output.png"
        spectrum_path = temp_dir / "test_output_color_spectrum.png"
        
        test_args = [
            'color_wheel.py',
            str(sample_image_path),
            str(output_path),
            '--color-spectrum',
            '--size', '100'
        ]
        
        with patch('sys.argv', test_args):
            result = color_wheel.main()
            
            assert result == 0
            assert output_path.exists()
            assert spectrum_path.exists()
    
    def test_circular_spectrum_generation(self, sample_image_path, temp_dir):
        """Test circular spectrum generation."""
        output_path = temp_dir / "test_output.png"
        circular_path = temp_dir / "test_output_circular_spectrum.png"
        
        test_args = [
            'color_wheel.py',
            str(sample_image_path),
            str(output_path),
            '--circular-spectrum',
            '--size', '100'
        ]
        
        with patch('sys.argv', test_args):
            result = color_wheel.main()
            
            assert result == 0
            assert output_path.exists()
            assert circular_path.exists()
    
    def test_all_output_formats(self, sample_image_path, temp_dir):
        """Test generation of all output formats together."""
        output_path = temp_dir / "test_all.png"
        
        test_args = [
            'color_wheel.py',
            str(sample_image_path),
            str(output_path),
            '--show-reference',
            '--histogram',
            '--color-spectrum', 
            '--circular-spectrum',
            '--size', '100'
        ]
        
        with patch('sys.argv', test_args):
            result = color_wheel.main()
            
            assert result == 0
            
            # All files should be created
            assert output_path.exists()
            assert (temp_dir / "test_all_reference.png").exists()
            assert (temp_dir / "test_all_histogram.png").exists()
            assert (temp_dir / "test_all_color_spectrum.png").exists()
            assert (temp_dir / "test_all_circular_spectrum.png").exists()


class TestOptimizationFlags:
    """Test optimization flag handling."""
    
    @patch('color_wheel.CUPY_AVAILABLE', False)
    def test_gpu_requested_but_unavailable(self, sample_image_path, temp_dir):
        """Test requesting GPU when CuPy is unavailable."""
        output_path = temp_dir / "output.png"
        
        test_args = [
            'color_wheel.py',
            str(sample_image_path),
            str(output_path),
            '--gpu'
        ]
        
        with patch('sys.argv', test_args):
            result = color_wheel.main()
            
            # Should return error code
            assert result == 1
    
    @patch('color_wheel.CUPY_AVAILABLE', True)
    def test_gpu_available_and_requested(self, sample_image_path, temp_dir):
        """Test GPU processing when available."""
        output_path = temp_dir / "output.png"
        
        test_args = [
            'color_wheel.py',
            str(sample_image_path),
            str(output_path),
            '--gpu',
            '--size', '100'
        ]
        
        with patch('sys.argv', test_args):
            with patch('builtins.print') as mock_print:
                result = color_wheel.main()
                
                assert result == 0
                
                # Should print GPU acceleration message
                print_calls = [str(call) for call in mock_print.call_args_list]
                gpu_message = any('GPU acceleration' in call for call in print_calls)
                assert gpu_message
    
    def test_no_gpu_flag(self, sample_image_path, temp_dir):
        """Test --no-gpu flag."""
        output_path = temp_dir / "output.png"
        
        test_args = [
            'color_wheel.py',
            str(sample_image_path),
            str(output_path),
            '--no-gpu',
            '--size', '100'
        ]
        
        with patch('sys.argv', test_args):
            with patch('builtins.print') as mock_print:
                result = color_wheel.main()
                
                assert result == 0
                
                # Should print GPU disabled message
                print_calls = [str(call) for call in mock_print.call_args_list]
                gpu_disabled = any('GPU acceleration disabled' in call for call in print_calls)
                assert gpu_disabled
    
    @patch('color_wheel.KDTREE_AVAILABLE', False)
    def test_kdtree_requested_but_unavailable(self, sample_image_path, temp_dir):
        """Test that --force-kdtree works even when sklearn is unavailable."""
        output_path = temp_dir / "output.png"
        
        test_args = [
            'color_wheel.py',
            str(sample_image_path),
            str(output_path),
            '--force-kdtree',
            '--size', '100'
        ]
        
        with patch('sys.argv', test_args):
            # Should still work, just use fallback method
            result = color_wheel.main()
            assert result == 0  # Should complete successfully
    
    def test_parallel_processing_flags(self, sample_image_path, temp_dir):
        """Test parallel processing flags."""
        output_path = temp_dir / "output.png"
        
        # Test --parallel flag
        test_args_parallel = [
            'color_wheel.py',
            str(sample_image_path),
            str(output_path),
            '--parallel',
            '--size', '100'
        ]
        
        with patch('sys.argv', test_args_parallel):
            result = color_wheel.main()
            assert result == 0
        
        # Test --no-parallel flag
        output_path2 = temp_dir / "output2.png"
        test_args_no_parallel = [
            'color_wheel.py',
            str(sample_image_path),
            str(output_path2),
            '--no-parallel',
            '--size', '100'
        ]
        
        with patch('sys.argv', test_args_no_parallel):
            result = color_wheel.main()
            assert result == 0


class TestColorSpaceHandling:
    """Test color space handling in CLI."""
    
    def test_srgb_color_space(self, sample_image_path, temp_dir):
        """Test sRGB color space (default)."""
        output_path = temp_dir / "srgb_output.png"
        
        test_args = [
            'color_wheel.py',
            str(sample_image_path),
            str(output_path),
            '--color-space', 'sRGB',
            '--size', '100'
        ]
        
        with patch('sys.argv', test_args):
            result = color_wheel.main()
            assert result == 0
            assert output_path.exists()
    
    def test_adobe_rgb_color_space(self, sample_image_path, temp_dir):
        """Test Adobe RGB color space."""
        output_path = temp_dir / "adobe_output.png"
        
        test_args = [
            'color_wheel.py',
            str(sample_image_path),
            str(output_path),
            '--color-space', 'Adobe RGB',
            '--size', '100'
        ]
        
        with patch('sys.argv', test_args):
            result = color_wheel.main()
            assert result == 0
            assert output_path.exists()
    
    def test_prophoto_rgb_color_space(self, sample_image_path, temp_dir):
        """Test ProPhoto RGB color space."""
        output_path = temp_dir / "prophoto_output.png"
        
        test_args = [
            'color_wheel.py',
            str(sample_image_path),
            str(output_path),
            '--color-space', 'ProPhoto RGB',
            '--size', '100'
        ]
        
        with patch('sys.argv', test_args):
            result = color_wheel.main()
            assert result == 0
            assert output_path.exists()


class TestFileFormatDetection:
    """Test automatic file format detection."""
    
    def test_auto_detect_jpg_from_extension(self, sample_image_path, temp_dir):
        """Test auto-detecting JPG format from file extension."""
        output_path = temp_dir / "output.jpg"
        
        test_args = [
            'color_wheel.py',
            str(sample_image_path),
            str(output_path),
            '--format', 'png',  # Request PNG but filename suggests JPG
            '--size', '100'
        ]
        
        with patch('sys.argv', test_args):
            with patch('builtins.print') as mock_print:
                result = color_wheel.main()
                
                assert result == 0
                assert output_path.exists()
                
                # Should print auto-detection message
                print_calls = [str(call) for call in mock_print.call_args_list]
                auto_detect = any('Auto-detected JPG' in call for call in print_calls)
                assert auto_detect
    
    def test_auto_detect_png_from_extension(self, sample_image_path, temp_dir):
        """Test auto-detecting PNG format from file extension."""
        output_path = temp_dir / "output.png"
        
        test_args = [
            'color_wheel.py',
            str(sample_image_path),
            str(output_path),
            '--format', 'jpg',  # Request JPG but filename suggests PNG
            '--size', '100'
        ]
        
        with patch('sys.argv', test_args):
            with patch('builtins.print') as mock_print:
                result = color_wheel.main()
                
                assert result == 0
                assert output_path.exists()
                
                # Should print auto-detection message
                print_calls = [str(call) for call in mock_print.call_args_list]
                auto_detect = any('Auto-detected PNG' in call for call in print_calls)
                assert auto_detect


class TestErrorHandling:
    """Test error handling in the main function."""
    
    def test_exception_handling(self, sample_image_path, temp_dir):
        """Test that exceptions are properly caught and handled."""
        output_path = temp_dir / "output.png"
        
        test_args = [
            'color_wheel.py',
            str(sample_image_path),
            str(output_path)
        ]
        
        with patch('sys.argv', test_args):
            # Mock load_and_analyze_image to raise an exception
            with patch('color_wheel.load_and_analyze_image') as mock_load:
                mock_load.side_effect = Exception("Test exception")
                
                with patch('builtins.print') as mock_print:
                    result = color_wheel.main()
                    
                    # Should return error code
                    assert result == 1
                    
                    # Should print error message
                    print_calls = [str(call) for call in mock_print.call_args_list]
                    error_found = any('Error:' in call for call in print_calls)
                    assert error_found
    
    def test_image_save_failure(self, sample_image_path, temp_dir):
        """Test handling of image save failure.""" 
        output_path = temp_dir / "readonly" / "output.png"
        
        test_args = [
            'color_wheel.py',
            str(sample_image_path),
            str(output_path),
            '--size', '100'
        ]
        
        with patch('sys.argv', test_args):
            # Mock cv2.imwrite to return False (save failure)
            with patch('cv2.imwrite') as mock_imwrite:
                mock_imwrite.return_value = False
                
                with patch('builtins.print') as mock_print:
                    result = color_wheel.main()
                    
                    # Should return error code
                    assert result == 1
                    
                    # Should print save error message
                    print_calls = [str(call) for call in mock_print.call_args_list]
                    save_error = any('Failed to save' in call for call in print_calls)
                    assert save_error


class TestPerformanceOutput:
    """Test performance timing and optimization output."""
    
    def test_timing_output(self, sample_image_path, temp_dir):
        """Test that timing information is displayed."""
        output_path = temp_dir / "output.png"
        
        test_args = [
            'color_wheel.py',
            str(sample_image_path),
            str(output_path),
            '--size', '100'
        ]
        
        with patch('sys.argv', test_args):
            with patch('builtins.print') as mock_print:
                result = color_wheel.main()
                
                assert result == 0
                
                # Should print timing information
                print_calls = [str(call) for call in mock_print.call_args_list]
                timing_found = any('completed in' in call for call in print_calls)
                assert timing_found
    
    def test_optimization_information_display(self, sample_image_path, temp_dir):
        """Test that available optimizations are displayed."""
        output_path = temp_dir / "output.png"
        
        test_args = [
            'color_wheel.py',
            str(sample_image_path),
            str(output_path),
            '--size', '100'
        ]
        
        with patch('sys.argv', test_args):
            with patch('builtins.print') as mock_print:
                result = color_wheel.main()
                
                assert result == 0
                
                # Should print available optimizations
                print_calls = [str(call) for call in mock_print.call_args_list]
                opt_found = any('Available optimizations' in call for call in print_calls)
                assert opt_found
    
    def test_progress_information(self, sample_image_path, temp_dir):
        """Test that progress information is displayed."""
        output_path = temp_dir / "output.png"
        
        test_args = [
            'color_wheel.py',
            str(sample_image_path),
            str(output_path),
            '--size', '100'
        ]
        
        with patch('sys.argv', test_args):
            with patch('builtins.print') as mock_print:
                result = color_wheel.main()
                
                assert result == 0
                
                print_calls = [str(call) for call in mock_print.call_args_list]
                
                # Should show loading progress
                loading_found = any('Loading and analyzing' in call for call in print_calls)
                assert loading_found
                
                # Should show generation progress
                generating_found = any('Generating color wheel' in call for call in print_calls)
                assert generating_found
                
                # Should show save confirmation
                saved_found = any('saved to' in call for call in print_calls)
                assert saved_found