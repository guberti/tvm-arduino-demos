import serial
import pygame

UPSCALE = 32

def main():
    parser = argparse.ArgumentParser(description='script to generate Arduino projects from .tflite models')
    parser.add_argument('port', type=str)
    parser.add_argument('--baudrate', type=int, default=115200)
    args = parser.parse_args()

    pygame.init()
    display = pygame.display.set_mode((32 * UPSCALE, 32 * UPSCALE))

    for attempts in range(10):
        if any(serial.tools.list_ports.grep(port)):
            break
        time.sleep(0.5)

    port = serial.Serial(args.port, baudrate=args.baudrate, timeout=5)

    while(True):
        if port.in_waiting >= 32 * 32 * 4:


if __name__ == '__main__':
    main()
