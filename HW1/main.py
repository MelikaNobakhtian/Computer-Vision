import cv2
imm = cv2.imread('background.png', cv2.IMREAD_COLOR)
imm_conv = cv2.cvtColor(imm, cv2.COLOR_BGR2RGB)
imm_conv = cv2.resize(imm_conv, (570, 290))
start_points = [(0, 0), (0, 0), (560, 280), (560, 280)]
end_points = [(0, 280), (560, 0), (560, 0), (0, 280)]
color = (0, 0, 255)
thickness = 3
for i in range(4):
    cv2.line(imm_conv, start_points[i], end_points[i], color, thickness)

centers = [(0, 0), (0, 280), (560, 0), (560, 280)]
colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (128, 0, 128)]
for center, colour in zip(centers, colors):
    cv2.circle(imm_conv, center, 3, colour, 5)


cv2.imshow('img', imm_conv)
cv2.waitKey(0)

# write end.png
new_img = cv2.cvtColor(imm, cv2.COLOR_BGR2RGB)
new_img = cv2.resize(new_img, (560, 290))
points = [(0, 145), (80, 145), (160, 145), (240, 145), (320, 145), (400, 145), (480, 145), (560, 145)]
heads = [(0, 0), (0, 290), (560, 0), (560, 290)]
color = (0, 0, 0)
for head in heads:
    for point in points:
        cv2.line(new_img, head, point, color, 1)

# final lines
cv2.line(new_img, (0, 0), (560, 0), color, 1)
cv2.line(new_img, (0, 0), (0, 290), color, 1)


cv2.imshow('img2', new_img)
cv2.waitKey(0)

cv2.imwrite('mypic.jpg', new_img)