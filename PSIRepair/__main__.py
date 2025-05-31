from .main import main, get_config

if __name__ == "__main__":
    config = get_config()
    exit(main(config))
