from psychopy import visual, core, event

# Create a simple window
win = visual.Window(size=(800, 600), color='black', fullscr=False)

# Show it for 3 seconds
for _ in range(180):  # 180 frames @ 60fps = 3 seconds
    win.flip()
    
win.close()
core.quit()
print("✅ Window test passed!")