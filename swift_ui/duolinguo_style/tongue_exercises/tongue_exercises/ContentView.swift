import SwiftUI

// Model for a word in the text with a unique ID for drag-and-drop
struct TextWord: Identifiable, Equatable {
    let id = UUID()
    let word: String
}

// Custom FlowLayout to wrap words like natural text
struct FlowLayout: Layout {
    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let sizes = subviews.map { $0.sizeThatFits(.unspecified) }
        let width = proposal.width ?? 0
        var height: CGFloat = 0
        var currentLineWidth: CGFloat = 0
        var currentLineHeight: CGFloat = 0
        
        for size in sizes {
            if currentLineWidth + size.width > width && currentLineWidth > 0 {
                height += currentLineHeight
                currentLineWidth = size.width
                currentLineHeight = size.height
            } else {
                currentLineWidth += size.width
                currentLineHeight = max(currentLineHeight, size.height)
            }
        }
        height += currentLineHeight
        return CGSize(width: width, height: height)
    }
    
    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let sizes = subviews.map { $0.sizeThatFits(.unspecified) }
        let width = bounds.width
        var x: CGFloat = bounds.minX
        var y: CGFloat = bounds.minY
        var currentLineHeight: CGFloat = 0
        
        for (index, subview) in subviews.enumerated() {
            let size = sizes[index]
            if x + size.width > bounds.maxX && x > bounds.minX {
                x = bounds.minX
                y += currentLineHeight
                currentLineHeight = 0
            }
            subview.place(at: CGPoint(x: x, y: y), proposal: .unspecified)
            x += size.width
            currentLineHeight = max(currentLineHeight, size.height)
        }
    }
}

struct ContentView: View {
    @State private var availableWords = ["Hello", "World", "Swift", "UI", "is", "awesome", "and", "fun"]
    @State private var composedText: [TextWord] = []
    
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            // Title
            Text("Language App")
                .font(.title)
                .padding(.horizontal)
            
            // Available Words
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(availableWords, id: \.self) { word in
                        Text(word)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.blue.opacity(0.2))
                            .cornerRadius(8)
                            .onTapGesture {
                                composedText.append(TextWord(word: word))
                            }
                    }
                }
                .padding(.horizontal)
            }
            
            // Composed Text Area
            ZStack {
                RoundedRectangle(cornerRadius: 10)
                    .fill(Color.gray.opacity(0.1))
                
                FlowLayout {
                    if composedText.isEmpty {
                        Text("Tap words to build your text")
                            .foregroundColor(.gray)
                            .padding()
                    } else {
                        ForEach(composedText) { textWord in
                            Text(textWord.word)
                                .padding(.horizontal, 4)
                                .padding(.vertical, 2)
                                .background(Color.green.opacity(0.2))
                                .cornerRadius(4)
                                .draggable(textWord.word) {
                                    Text(textWord.word)
                                        .padding(.horizontal, 4)
                                        .padding(.vertical, 2)
                                        .background(Color.green.opacity(0.5))
                                        .cornerRadius(4)
                                }
                                .dropDestination(for: String.self) { items, location in
                                    guard let draggedWord = items.first,
                                          let draggedIndex = composedText.firstIndex(where: { $0.word == draggedWord }),
                                          let targetIndex = composedText.firstIndex(where: { $0.id == textWord.id }) else {
                                        return false
                                    }
                                    let draggedTextWord = composedText[draggedIndex]
                                    composedText.remove(at: draggedIndex)
                                    composedText.insert(draggedTextWord, at: targetIndex)
                                    return true
                                }
                        }
                    }
                }
                .padding()
            }
            .padding(.horizontal)
            
            // Clear Button
            Button(action: {
                composedText.removeAll()
            }) {
                Text("Clear Text")
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.red.opacity(0.8))
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
            .padding(.horizontal)
            
            Spacer()
        }
        .padding(.vertical)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
