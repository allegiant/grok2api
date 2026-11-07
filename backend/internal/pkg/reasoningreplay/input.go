package reasoningreplay

import (
	"encoding/json"
	"strings"
)

func filterReplayItemsForInput(body []byte, items [][]byte) [][]byte {
	var payload struct {
		Input []json.RawMessage `json:"input"`
	}
	if json.Unmarshal(body, &payload) != nil || len(payload.Input) == 0 {
		return nil
	}
	inputItems := make([]map[string]json.RawMessage, 0, len(payload.Input))
	for _, raw := range payload.Input {
		var item map[string]json.RawMessage
		if json.Unmarshal(raw, &item) == nil {
			inputItems = append(inputItems, item)
		}
	}
	lastAssistant, hasLastAssistant := lastAssistantMessage(inputItems)
	cachedAssistant, hasCachedAssistant := replayAssistantMessage(items)
	assistantMatches := hasLastAssistant && hasCachedAssistant && assistantContentEqual(lastAssistant, cachedAssistant)
	existingCalls := map[string]bool{}
	existingOutputs := map[string]string{}
	existingEncrypted := map[string]bool{}
	for _, item := range inputItems {
		var typeName string
		_ = json.Unmarshal(item["type"], &typeName)
		typeName = strings.TrimSpace(typeName)
		switch typeName {
		case "reasoning":
			var enc string
			_ = json.Unmarshal(item["encrypted_content"], &enc)
			if enc != "" {
				existingEncrypted[enc] = true
			}
		case "function_call_output", "custom_tool_call_output":
			var callID string
			_ = json.Unmarshal(item["call_id"], &callID)
			for _, candidate := range comparableReplayCallIDs(callID) {
				existingOutputs[candidate] = callID
			}
		case "function_call", "custom_tool_call":
			var callID string
			_ = json.Unmarshal(item["call_id"], &callID)
			for _, key := range replayToolCallKeys(typeName, callID) {
				existingCalls[key] = true
			}
		}
	}
	hasMatchedToolCall := false
	for _, item := range items {
		var typed struct {
			Type   string `json:"type"`
			CallID string `json:"call_id"`
		}
		if json.Unmarshal(item, &typed) != nil {
			continue
		}
		typeName := strings.TrimSpace(typed.Type)
		if typeName != "function_call" && typeName != "custom_tool_call" {
			continue
		}
		keys := replayToolCallKeys(typeName, typed.CallID)
		if len(keys) > 0 && anyReplayCallKeyExists(existingCalls, keys) {
			hasMatchedToolCall = true
			break
		}
		for _, candidate := range comparableReplayCallIDs(typed.CallID) {
			if existingOutputs[candidate] != "" {
				hasMatchedToolCall = true
				break
			}
		}
		if hasMatchedToolCall {
			break
		}
	}
	if hasLastAssistant && hasCachedAssistant && !assistantMatches && !hasMatchedToolCall {
		return nil
	}
	filtered := make([][]byte, 0, len(items))
	for _, item := range items {
		var typed struct {
			Type             string `json:"type"`
			EncryptedContent string `json:"encrypted_content"`
			CallID           string `json:"call_id"`
		}
		if json.Unmarshal(item, &typed) != nil {
			continue
		}
		switch strings.TrimSpace(typed.Type) {
		case "reasoning":
			if existingEncrypted[typed.EncryptedContent] {
				continue
			}
		case "message":
			if assistantMatches || hasLastAssistant {
				continue
			}
		case "function_call", "custom_tool_call":
			keys := replayToolCallKeys(strings.TrimSpace(typed.Type), typed.CallID)
			if len(keys) == 0 || anyReplayCallKeyExists(existingCalls, keys) {
				continue
			}
			outputCallID := ""
			for _, candidate := range comparableReplayCallIDs(typed.CallID) {
				if value := existingOutputs[candidate]; value != "" {
					outputCallID = value
					break
				}
			}
			if outputCallID == "" {
				continue
			}
			for _, key := range keys {
				existingCalls[key] = true
			}
			if outputCallID != typed.CallID {
				item = rewriteReplayCallID(item, outputCallID)
			}
		default:
			continue
		}
		filtered = append(filtered, item)
	}
	return filtered
}

func comparableReplayCallIDs(callID string) []string {
	callID = strings.TrimSpace(callID)
	if callID == "" {
		return nil
	}
	const anthropicPrefix = "toolu_"
	if strings.HasPrefix(callID, anthropicPrefix) {
		upstreamID := strings.TrimPrefix(callID, anthropicPrefix)
		if upstreamID != "" {
			return []string{callID, upstreamID}
		}
		return []string{callID}
	}
	return []string{callID, anthropicPrefix + callID}
}

func replayToolCallKeys(itemType, callID string) []string {
	if itemType != "function_call" && itemType != "custom_tool_call" {
		return nil
	}
	return comparableReplayCallIDs(callID)
}

func anyReplayCallKeyExists(existing map[string]bool, keys []string) bool {
	for _, key := range keys {
		if existing[key] {
			return true
		}
	}
	return false
}

func rewriteReplayCallID(item []byte, callID string) []byte {
	var raw map[string]json.RawMessage
	if json.Unmarshal(item, &raw) != nil {
		return item
	}
	encoded, err := json.Marshal(callID)
	if err != nil {
		return item
	}
	raw["call_id"] = encoded
	updated, err := json.Marshal(raw)
	if err != nil {
		return item
	}
	return updated
}

func isAssistantMessageItem(item map[string]json.RawMessage) bool {
	if len(item) == 0 {
		return false
	}
	var typeName, role string
	_ = json.Unmarshal(item["type"], &typeName)
	_ = json.Unmarshal(item["role"], &role)
	typeName = strings.TrimSpace(typeName)
	return (typeName == "" || typeName == "message") && strings.EqualFold(strings.TrimSpace(role), "assistant")
}

func isPortableReasoningSummaryItem(item map[string]json.RawMessage) bool {
	if !isAssistantMessageItem(item) {
		return false
	}
	parts, ok := assistantParts(item["content"])
	if !ok || len(parts) != 1 {
		return false
	}
	return strings.HasPrefix(strings.TrimSpace(parts[0].value), "Prior model reasoning summary:")
}

func lastAssistantMessage(items []map[string]json.RawMessage) (map[string]json.RawMessage, bool) {
	for index := len(items) - 1; index >= 0; index-- {
		item := items[index]
		if !isAssistantMessageItem(item) || isPortableReasoningSummaryItem(item) {
			continue
		}
		return item, true
	}
	return nil, false
}

func replayAssistantMessage(items [][]byte) (map[string]json.RawMessage, bool) {
	for _, item := range items {
		var raw map[string]json.RawMessage
		if json.Unmarshal(item, &raw) != nil {
			continue
		}
		var typeName, role string
		_ = json.Unmarshal(raw["type"], &typeName)
		_ = json.Unmarshal(raw["role"], &role)
		if strings.TrimSpace(typeName) == "message" && strings.EqualFold(strings.TrimSpace(role), "assistant") {
			return raw, true
		}
	}
	return nil, false
}

func assistantContentEqual(left, right map[string]json.RawMessage) bool {
	leftParts, leftOK := assistantParts(left["content"])
	rightParts, rightOK := assistantParts(right["content"])
	if !leftOK || !rightOK || len(leftParts) != len(rightParts) {
		return false
	}
	for i := range leftParts {
		if leftParts[i] != rightParts[i] {
			return false
		}
	}
	return true
}

type assistantPart struct {
	partType string
	value    string
}

func assistantParts(raw json.RawMessage) ([]assistantPart, bool) {
	if len(raw) == 0 {
		return nil, false
	}
	var asString string
	if json.Unmarshal(raw, &asString) == nil {
		return []assistantPart{{partType: "output_text", value: asString}}, true
	}
	var parts []map[string]json.RawMessage
	if json.Unmarshal(raw, &parts) != nil {
		return nil, false
	}
	result := make([]assistantPart, 0, len(parts))
	for _, part := range parts {
		var partType string
		_ = json.Unmarshal(part["type"], &partType)
		switch strings.TrimSpace(partType) {
		case "output_text":
			var text string
			if json.Unmarshal(part["text"], &text) != nil {
				return nil, false
			}
			result = append(result, assistantPart{partType: partType, value: text})
		case "refusal":
			var refusal string
			if json.Unmarshal(part["refusal"], &refusal) != nil {
				return nil, false
			}
			result = append(result, assistantPart{partType: partType, value: refusal})
		default:
			return nil, false
		}
	}
	return result, len(result) > 0
}

func insertReplayItems(body []byte, replayItems [][]byte, allCachedItems ...[][]byte) ([]byte, bool) {
	var payload map[string]json.RawMessage
	if json.Unmarshal(body, &payload) != nil {
		return body, false
	}
	inputRaw, ok := payload["input"]
	if !ok {
		return body, false
	}
	var input []json.RawMessage
	if json.Unmarshal(inputRaw, &input) != nil {
		return body, false
	}
	insertAt := replayInsertIndex(input, replayItems, allCachedItems...)
	// 不基于不可信请求长度计算预分配容量，交由 append 安全扩容。
	next := make([]json.RawMessage, 0)
	for i, item := range input {
		if i == insertAt {
			for _, replay := range replayItems {
				next = append(next, json.RawMessage(replay))
			}
		}
		next = append(next, item)
	}
	if insertAt == len(input) {
		for _, replay := range replayItems {
			next = append(next, json.RawMessage(replay))
		}
	}
	encoded, err := json.Marshal(next)
	if err != nil {
		return body, false
	}
	payload["input"] = encoded
	updated, err := json.Marshal(payload)
	if err != nil {
		return body, false
	}
	return updated, true
}

func collectReplayCallIDs(items [][]byte, dst map[string]bool) {
	for _, item := range items {
		var typed struct {
			Type   string `json:"type"`
			CallID string `json:"call_id"`
		}
		if json.Unmarshal(item, &typed) != nil {
			continue
		}
		if typed.Type == "function_call" || typed.Type == "custom_tool_call" {
			for _, id := range comparableReplayCallIDs(typed.CallID) {
				dst[id] = true
			}
		}
	}
}

func replayInsertIndex(input []json.RawMessage, replayItems [][]byte, allCachedItems ...[][]byte) int {
	replayCallIDs := map[string]bool{}
	collectReplayCallIDs(replayItems, replayCallIDs)
	for _, group := range allCachedItems {
		collectReplayCallIDs(group, replayCallIDs)
	}
	if len(replayCallIDs) > 0 {
		for index, raw := range input {
			var typed struct {
				Type   string `json:"type"`
				CallID string `json:"call_id"`
			}
			if json.Unmarshal(raw, &typed) != nil {
				continue
			}
			typeName := strings.TrimSpace(typed.Type)
			isCall := typeName == "function_call" || typeName == "custom_tool_call"
			isOutput := typeName == "function_call_output" || typeName == "custom_tool_call_output"
			if !isCall && !isOutput {
				continue
			}
			callIDs := comparableReplayCallIDs(typed.CallID)
			if len(callIDs) == 0 {
				if isOutput {
					return index
				}
				continue
			}
			for _, callID := range callIDs {
				if replayCallIDs[callID] {
					return index
				}
			}
		}
	}
	for index := len(input) - 1; index >= 0; index-- {
		var typed struct {
			Type string `json:"type"`
			Role string `json:"role"`
		}
		if json.Unmarshal(input[index], &typed) != nil {
			continue
		}
		typeName := strings.TrimSpace(typed.Type)
		if (typeName == "" || typeName == "message") && strings.EqualFold(strings.TrimSpace(typed.Role), "assistant") {
			return index
		}
	}
	for index, raw := range input {
		if shouldInsertReplayBefore(raw) {
			return index
		}
	}
	return len(input)
}

func shouldInsertReplayBefore(raw json.RawMessage) bool {
	var typed struct {
		Type string `json:"type"`
		Role string `json:"role"`
	}
	if json.Unmarshal(raw, &typed) != nil {
		return true
	}
	typeName := strings.TrimSpace(typed.Type)
	role := strings.ToLower(strings.TrimSpace(typed.Role))
	if role == "" || (typeName != "" && typeName != "message") {
		return true
	}
	return role != "system" && role != "developer"
}

func splitReplayTurnGroups(items [][]byte) [][][]byte {
	if len(items) == 0 {
		return nil
	}
	groups := make([][][]byte, 0, 4)
	var current [][]byte
	seenNonReasoning := false
	for _, item := range items {
		var typed struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(item, &typed) != nil {
			continue
		}
		typeName := strings.TrimSpace(typed.Type)
		if typeName == "" {
			continue
		}
		if typeName == "reasoning" && seenNonReasoning && len(current) > 0 {
			groups = append(groups, current)
			current = nil
			seenNonReasoning = false
		}
		if typeName != "reasoning" {
			seenNonReasoning = true
		}
		current = append(current, append([]byte(nil), item...))
	}
	if len(current) > 0 {
		groups = append(groups, current)
	}
	return groups
}

func groupHasReasoning(group [][]byte) bool {
	for _, item := range group {
		var typed struct {
			Type             string `json:"type"`
			EncryptedContent string `json:"encrypted_content"`
		}
		if json.Unmarshal(item, &typed) == nil && strings.TrimSpace(typed.Type) == "reasoning" && strings.TrimSpace(typed.EncryptedContent) != "" {
			return true
		}
	}
	return false
}

func isAnchoredReplayGroup(group [][]byte) bool {
	hasReasoning := false
	hasAnchor := false
	for _, item := range group {
		var typed struct {
			Type             string `json:"type"`
			EncryptedContent string `json:"encrypted_content"`
		}
		if json.Unmarshal(item, &typed) != nil {
			continue
		}
		switch strings.TrimSpace(typed.Type) {
		case "reasoning":
			if strings.TrimSpace(typed.EncryptedContent) != "" {
				hasReasoning = true
			}
		case "function_call", "custom_tool_call", "message":
			hasAnchor = true
		}
	}
	return hasReasoning && hasAnchor
}

func replayGroupOverlap(left, right [][]byte) bool {
	leftEncrypted := map[string]bool{}
	leftCalls := map[string]bool{}
	for _, item := range left {
		var typed struct {
			Type             string `json:"type"`
			EncryptedContent string `json:"encrypted_content"`
			CallID           string `json:"call_id"`
		}
		if json.Unmarshal(item, &typed) != nil {
			continue
		}
		switch strings.TrimSpace(typed.Type) {
		case "reasoning":
			if enc := strings.TrimSpace(typed.EncryptedContent); enc != "" {
				leftEncrypted[enc] = true
			}
		case "function_call", "custom_tool_call":
			for _, id := range comparableReplayCallIDs(typed.CallID) {
				leftCalls[id] = true
			}
		}
	}
	for _, item := range right {
		var typed struct {
			Type             string `json:"type"`
			EncryptedContent string `json:"encrypted_content"`
			CallID           string `json:"call_id"`
		}
		if json.Unmarshal(item, &typed) != nil {
			continue
		}
		switch strings.TrimSpace(typed.Type) {
		case "reasoning":
			if enc := strings.TrimSpace(typed.EncryptedContent); enc != "" && leftEncrypted[enc] {
				return true
			}
		case "function_call", "custom_tool_call":
			for _, id := range comparableReplayCallIDs(typed.CallID) {
				if leftCalls[id] {
					return true
				}
			}
		}
	}
	return false
}

func mergeReplayTurnGroups(existing, normalized [][]byte, normalizedOK bool, maxGroups int) [][]byte {
	existingGroups := splitReplayTurnGroups(existing)
	kept := make([][][]byte, 0, len(existingGroups)+1)
	for _, group := range existingGroups {
		if isAnchoredReplayGroup(group) {
			kept = append(kept, group)
		}
	}
	if normalizedOK && groupHasReasoning(normalized) {
		for _, newGroup := range splitReplayTurnGroups(normalized) {
			if !groupHasReasoning(newGroup) {
				continue
			}
			filteredKept := make([][][]byte, 0, len(kept)+1)
			for _, prior := range kept {
				if !replayGroupOverlap(prior, newGroup) {
					filteredKept = append(filteredKept, prior)
				}
			}
			kept = append(filteredKept, newGroup)
		}
	}
	if maxGroups > 0 && len(kept) > maxGroups {
		kept = kept[len(kept)-maxGroups:]
	}
	total := 0
	for _, group := range kept {
		total += len(group)
	}
	if total == 0 {
		return nil
	}
	merged := make([][]byte, 0, total)
	for _, group := range kept {
		merged = append(merged, group...)
	}
	return merged
}

func applyReplayItemsToBody(body []byte, items [][]byte) ([]byte, int, string) {
	groups := splitReplayTurnGroups(items)
	if len(groups) == 0 {
		return body, 0, "filtered"
	}
	if len(groups) == 1 {
		if updated, injected, matched := injectAnchoredReplayGroup(body, groups[0]); matched {
			if injected == 0 {
				return body, 0, "filtered"
			}
			return updated, injected, ""
		}
		filtered := filterReplayItemsForInput(body, groups[0])
		if len(filtered) == 0 {
			return body, 0, "filtered"
		}
		updated, ok := insertReplayItems(body, filtered, groups[0])
		if !ok {
			return body, 0, "insert_failed"
		}
		return updated, len(filtered), ""
	}
	current := body
	totalInjected := 0
	for index, group := range groups {
		updated, injected, matched := injectAnchoredReplayGroup(current, group)
		if matched {
			if injected > 0 {
				current = updated
				totalInjected += injected
			}
			continue
		}
		if index == len(groups)-1 && !hasInputAssistantOrToolItems(current) {
			filtered := filterReplayItemsForInput(current, group)
			if len(filtered) > 0 {
				if fallbackUpdated, ok := insertReplayItems(current, filtered, group); ok {
					current = fallbackUpdated
					totalInjected += len(filtered)
				}
			}
		}
	}
	if totalInjected == 0 {
		return body, 0, "filtered"
	}
	return current, totalInjected, ""
}

func hasInputAssistantOrToolItems(body []byte) bool {
	var payload struct {
		Input []map[string]json.RawMessage `json:"input"`
	}
	if json.Unmarshal(body, &payload) != nil {
		return false
	}
	for _, item := range payload.Input {
		if isAssistantMessageItem(item) && !isPortableReasoningSummaryItem(item) {
			return true
		}
		var typeName string
		_ = json.Unmarshal(item["type"], &typeName)
		switch strings.TrimSpace(typeName) {
		case "function_call", "custom_tool_call", "function_call_output", "custom_tool_call_output":
			return true
		}
	}
	return false
}

func isEncryptedReasoningItem(item map[string]json.RawMessage) bool {
	if len(item) == 0 {
		return false
	}
	var typeName, enc string
	_ = json.Unmarshal(item["type"], &typeName)
	_ = json.Unmarshal(item["encrypted_content"], &enc)
	return strings.TrimSpace(typeName) == "reasoning" && strings.TrimSpace(enc) != ""
}

func injectAnchoredReplayGroup(body []byte, group [][]byte) ([]byte, int, bool) {
	var payload map[string]json.RawMessage
	if json.Unmarshal(body, &payload) != nil {
		return body, 0, false
	}
	inputRaw, ok := payload["input"]
	if !ok {
		return body, 0, false
	}
	var rawInput []json.RawMessage
	if json.Unmarshal(inputRaw, &rawInput) != nil || len(rawInput) == 0 {
		return body, 0, false
	}
	inputItems := make([]map[string]json.RawMessage, len(rawInput))
	existingEncrypted := map[string]bool{}
	existingCalls := map[string]bool{}
	existingOutputs := map[string]string{}
	for i, raw := range rawInput {
		var item map[string]json.RawMessage
		if json.Unmarshal(raw, &item) != nil {
			continue
		}
		inputItems[i] = item
		var typeName string
		_ = json.Unmarshal(item["type"], &typeName)
		typeName = strings.TrimSpace(typeName)
		switch typeName {
		case "reasoning":
			var enc string
			_ = json.Unmarshal(item["encrypted_content"], &enc)
			if enc = strings.TrimSpace(enc); enc != "" {
				existingEncrypted[enc] = true
			}
		case "function_call", "custom_tool_call":
			var callID string
			_ = json.Unmarshal(item["call_id"], &callID)
			for _, key := range replayToolCallKeys(typeName, callID) {
				existingCalls[key] = true
			}
		case "function_call_output", "custom_tool_call_output":
			var callID string
			_ = json.Unmarshal(item["call_id"], &callID)
			for _, candidate := range comparableReplayCallIDs(callID) {
				existingOutputs[candidate] = callID
			}
		}
	}

	groupCallIDs := map[string]bool{}
	collectReplayCallIDs(group, groupCallIDs)
	cachedAssistant, hasCachedAssistant := replayAssistantMessage(group)

	insertAt := -1
	matchedByToolCall := false
	if len(groupCallIDs) > 0 {
		for idx, item := range inputItems {
			if item == nil {
				continue
			}
			var typeName, callID string
			_ = json.Unmarshal(item["type"], &typeName)
			_ = json.Unmarshal(item["call_id"], &callID)
			typeName = strings.TrimSpace(typeName)
			if typeName != "function_call" && typeName != "custom_tool_call" && typeName != "function_call_output" && typeName != "custom_tool_call_output" {
				continue
			}
			for _, candidate := range comparableReplayCallIDs(callID) {
				if groupCallIDs[candidate] {
					insertAt = idx
					matchedByToolCall = true
					break
				}
			}
			if matchedByToolCall {
				break
			}
		}
	}
	if matchedByToolCall {
		if hasCachedAssistant && insertAt > 0 && inputItems[insertAt-1] != nil && isAssistantMessageItem(inputItems[insertAt-1]) && assistantContentEqual(inputItems[insertAt-1], cachedAssistant) {
			insertAt--
		}
	} else if len(groupCallIDs) == 0 && hasCachedAssistant {
		for idx, item := range inputItems {
			if item == nil || !isAssistantMessageItem(item) || isPortableReasoningSummaryItem(item) {
				continue
			}
			if !assistantContentEqual(item, cachedAssistant) {
				continue
			}
			if insertAt < 0 {
				insertAt = idx
			}
			if idx == 0 || !isEncryptedReasoningItem(inputItems[idx-1]) {
				insertAt = idx
				break
			}
		}
	}
	if insertAt < 0 {
		return body, 0, false
	}

	toInsert := make([][]byte, 0, len(group))
	for _, item := range group {
		var typed struct {
			Type             string `json:"type"`
			EncryptedContent string `json:"encrypted_content"`
			CallID           string `json:"call_id"`
		}
		if json.Unmarshal(item, &typed) != nil {
			continue
		}
		typeName := strings.TrimSpace(typed.Type)
		switch typeName {
		case "reasoning":
			enc := strings.TrimSpace(typed.EncryptedContent)
			if enc == "" || existingEncrypted[enc] {
				continue
			}
			existingEncrypted[enc] = true
			toInsert = append(toInsert, item)
		case "function_call", "custom_tool_call":
			if !matchedByToolCall {
				continue
			}
			keys := replayToolCallKeys(typeName, typed.CallID)
			if len(keys) == 0 || anyReplayCallKeyExists(existingCalls, keys) {
				continue
			}
			outputCallID := ""
			for _, candidate := range comparableReplayCallIDs(typed.CallID) {
				if value := existingOutputs[candidate]; value != "" {
					outputCallID = value
					break
				}
			}
			if outputCallID == "" {
				continue
			}
			for _, key := range keys {
				existingCalls[key] = true
			}
			if outputCallID != typed.CallID {
				item = rewriteReplayCallID(item, outputCallID)
			}
			toInsert = append(toInsert, item)
		}
	}
	if len(toInsert) == 0 {
		return body, 0, true
	}

	replacePriorSummary := insertAt > 0 && inputItems[insertAt-1] != nil && isPortableReasoningSummaryItem(inputItems[insertAt-1])
	next := make([]json.RawMessage, 0, len(rawInput)+len(toInsert))
	for i, raw := range rawInput {
		if replacePriorSummary && i == insertAt-1 {
			for _, replay := range toInsert {
				next = append(next, json.RawMessage(replay))
			}
			continue
		}
		if !replacePriorSummary && i == insertAt {
			for _, replay := range toInsert {
				next = append(next, json.RawMessage(replay))
			}
		}
		next = append(next, raw)
	}
	encoded, err := json.Marshal(next)
	if err != nil {
		return body, 0, false
	}
	payload["input"] = encoded
	updated, err := json.Marshal(payload)
	if err != nil {
		return body, 0, false
	}
	return updated, len(toInsert), true
}

// CaptureBody 包装上游响应，在流结束时抽取并写入回放缓存。
