"""
Simple Python script for testing face swap directly
"""

from face_swapper import MultiFaceSwapper, load_image, save_image
import sys


def main():
    if len(sys.argv) < 4:
        print("Usage: python simple_swap.py <base_image> <face1> <face2> [face3] [face4]")
        print("\nExample:")
        print("  python simple_swap.py base.jpg face1.jpg face2.jpg")
        print("\nThis will:")
        print("  - Load base.jpg (should contain 2+ faces)")
        print("  - Replace faces with face1.jpg and face2.jpg (left to right)")
        print("  - Save result as output.jpg")
        sys.exit(1)
    
    # Parse arguments
    base_image_path = sys.argv[1]
    face_paths = sys.argv[2:]
    
    if len(face_paths) > 4:
        print("Error: Maximum 4 faces can be swapped at once")
        sys.exit(1)
    
    print(f"🎭 Multi-Face Swapper")
    print(f"=" * 50)
    print(f"Base image: {base_image_path}")
    print(f"Source faces: {len(face_paths)}")
    for i, path in enumerate(face_paths, 1):
        print(f"  Face {i}: {path}")
    print()
    
    # Initialize swapper
    print("Initializing face swapper...")
    swapper = MultiFaceSwapper()
    print("✓ Face swapper ready\n")
    
    # Load images
    print("Loading images...")
    try:
        base_image = load_image(base_image_path)
        face_images = [load_image(path) for path in face_paths]
        print("✓ All images loaded\n")
    except Exception as e:
        print(f"✗ Error loading images: {e}")
        sys.exit(1)
    
    # Perform face swap
    print("Swapping faces...")
    try:
        result_image, swap_info = swapper.swap_faces(base_image, face_images)
        print("✓ Face swap completed\n")
        
        # Print swap information
        print("Swap Information:")
        print(f"  Total faces in base image: {swap_info['total_base_faces']}")
        print(f"  Faces swapped: {swap_info['swapped_count']}")
        for i, swap in enumerate(swap_info['swaps'], 1):
            print(f"  Face {i} -> Target index {swap['target_index']}")
        print()
        
        # Save result
        output_path = "output.jpg"
        save_image(result_image, output_path)
        print(f"✓ Result saved to: {output_path}")
        
    except Exception as e:
        print(f"✗ Error during face swap: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
