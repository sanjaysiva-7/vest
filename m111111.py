from ultralytics import YOLO, checks, hub

def main():
    checks()
    
    hub.login('ad277e7eb9d1158634f554b19d04692c0e15524815')

    model = YOLO('b.ultralytics.com/models/YLA5BuxfsddkvBOY3611')
    results = model.train()
    
    return results

if __name__ == "__main__":
    main()
