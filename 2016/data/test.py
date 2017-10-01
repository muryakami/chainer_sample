test_err = [list(map(float, input().split()))[3] for i in range(500)]

print(*test_err)
print(len(test_err))
print(len(test_err[100:]))

sum = 0
for i in test_err[100:]:
    sum += i
print(sum/400)
