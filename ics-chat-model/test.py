import argparse

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('input', type=str, help='The input message')
	args = parser.parse_args()
	print(args.input)

if __name__ == "__main__":
    main()
